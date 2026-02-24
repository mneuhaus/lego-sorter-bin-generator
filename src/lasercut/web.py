"""Small web UI for generating lasercut SVG files."""

from __future__ import annotations

import html
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import cadquery as cq
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from lasercut.exporter import export_svg
from lasercut.joints import apply_finger_joints
from lasercut.panels import load_step_panels


_JOB_LOCK = Lock()
_JOB_INDEX: dict[str, dict[str, Any]] = {}
_JOB_TTL_SECONDS = int(float(os.getenv("LASERCUT_WEB_JOB_TTL_SECONDS", "21600")))
_JOB_MAX_WORKERS = max(1, int(float(os.getenv("LASERCUT_WEB_MAX_WORKERS", "4"))))
_FIXED_PACK_ROTATIONS = 8
_PREVIEW_LOCK = Lock()
_STEP_PREVIEW_CACHE: dict[str, dict[str, Any] | None] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _step_dir() -> Path:
    configured = os.getenv("LASERCUT_STEP_DIR")
    if configured:
        return Path(configured)
    return _repo_root() / "step_files"


def _job_root() -> Path:
    root = Path(tempfile.gettempdir()) / "lasercut-web-jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _num_token(value: float) -> str:
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    return s or "0"


def _available_step_files() -> list[str]:
    step_dir = _step_dir()
    if not step_dir.exists():
        return []
    return sorted(
        p.name
        for p in step_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".step", ".stp"}
    )


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _folder_and_filename(
    step_stem: str,
    layout: str,
    thickness: float,
    kerf: float,
    sheet_width: float | None,
    sheet_height: float | None,
) -> tuple[str, str]:
    thickness_label = f"{_num_token(thickness)}mm"
    kerf_label = f"k{_num_token(kerf)}mm"

    if layout == "packed":
        sw = int(round(sheet_width or 0))
        sh = int(round(sheet_height or 0))
        folder_name = f"bins_{thickness_label}_{kerf_label}_{sw}x{sh}"
        filename = f"{step_stem}-packed-{thickness_label}-{kerf_label}-{sw}x{sh}.svg"
    else:
        folder_name = f"bins_{thickness_label}_{kerf_label}"
        filename = f"{step_stem}-unfolded-{thickness_label}-{kerf_label}.svg"

    return folder_name, filename


def _cleanup_expired_jobs() -> None:
    now = time.time()
    expired: list[str] = []

    with _JOB_LOCK:
        for job_id, entry in _JOB_INDEX.items():
            created = float(entry.get("created_at", 0.0))
            if now - created > _JOB_TTL_SECONDS:
                expired.append(job_id)

        for job_id in expired:
            entry = _JOB_INDEX.pop(job_id, None)
            if not entry:
                continue
            job_dir = Path(entry["job_dir"])
            shutil.rmtree(job_dir, ignore_errors=True)


def _to_xyz(v: Any) -> tuple[float, float, float]:
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return (float(v.x), float(v.y), float(v.z))
    if hasattr(v, "toTuple"):
        t = v.toTuple()
        return (float(t[0]), float(t[1]), float(t[2]))
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    raise ValueError(f"Unsupported vector type: {type(v)}")


def _mesh_from_step(step_path: Path, tess_tol: float = 1.0) -> dict[str, Any] | None:
    """Create a simple triangle-mesh payload for browser 3D previews."""
    try:
        imported = cq.importers.importStep(str(step_path))
    except Exception:
        return None

    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    idx_offset = 0

    min_x = float("inf")
    min_y = float("inf")
    min_z = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    max_z = float("-inf")

    for obj in imported.objects:
        try:
            verts_raw, tris_raw = obj.tessellate(tess_tol)
        except Exception:
            continue

        if not verts_raw or not tris_raw:
            continue

        local_vertices: list[list[float]] = []
        for vert in verts_raw:
            x, y, z = _to_xyz(vert)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)
            local_vertices.append([round(x, 4), round(y, 4), round(z, 4)])

        vertices.extend(local_vertices)

        for tri in tris_raw:
            a, b, c = tri
            triangles.append([int(a) + idx_offset, int(b) + idx_offset, int(c) + idx_offset])

        idx_offset += len(local_vertices)

    if not vertices or not triangles:
        return None

    return {
        "vertices": vertices,
        "triangles": triangles,
        "bounds": {
            "min": [round(min_x, 4), round(min_y, 4), round(min_z, 4)],
            "max": [round(max_x, 4), round(max_y, 4), round(max_z, 4)],
        },
    }


def _get_step_preview_mesh(step_file: str) -> dict[str, Any] | None:
    with _PREVIEW_LOCK:
        if step_file in _STEP_PREVIEW_CACHE:
            return _STEP_PREVIEW_CACHE[step_file]

    mesh = _mesh_from_step(_step_dir() / step_file)

    with _PREVIEW_LOCK:
        _STEP_PREVIEW_CACHE[step_file] = mesh
    return mesh


def _generate_single_file(
    step_file: str,
    *,
    thickness: float,
    finger_width: float,
    kerf: float,
    layout: str,
    sheet_width: float | None,
    sheet_height: float | None,
    part_gap: float,
    sheet_gap: float,
    job_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    step_path = _step_dir() / step_file

    try:
        original_model = load_step_panels(str(step_path), thickness)
        model = apply_finger_joints(original_model, finger_width=finger_width, kerf=kerf)

        folder_name, filename = _folder_and_filename(
            step_stem=step_path.stem,
            layout=layout,
            thickness=thickness,
            kerf=kerf,
            sheet_width=sheet_width,
            sheet_height=sheet_height,
        )

        out_dir = job_dir / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / filename

        export_svg(
            model,
            str(output_path),
            reference_model=original_model,
            layout=layout,
            sheet_width=sheet_width,
            sheet_height=sheet_height,
            part_gap=part_gap,
            sheet_gap=sheet_gap,
            pack_rotations=_FIXED_PACK_ROTATIONS,
        )

        elapsed = round(time.perf_counter() - started, 2)
        return {
            "step_file": step_file,
            "ok": True,
            "filename": filename,
            "folder_name": folder_name,
            "output_path": str(output_path),
            "elapsed_s": elapsed,
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 2)
        return {
            "step_file": step_file,
            "ok": False,
            "error": str(exc),
            "elapsed_s": elapsed,
        }


def _run_batch_generation(
    step_files: list[str],
    *,
    thickness: float,
    finger_width: float,
    kerf: float,
    layout: str,
    sheet_width: float | None,
    sheet_height: float | None,
    part_gap: float,
    sheet_gap: float,
) -> dict[str, Any]:
    _cleanup_expired_jobs()

    available = set(_available_step_files())
    unknown = [name for name in step_files if name not in available]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown step file(s): {', '.join(sorted(unknown))}")

    deduped: list[str] = []
    seen: set[str] = set()
    for name in step_files:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    if not deduped:
        raise HTTPException(status_code=400, detail="No STEP files selected")

    job_id = f"{int(time.time())}-{os.urandom(4).hex()}"
    job_dir = _job_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    max_workers = min(len(deduped), _JOB_MAX_WORKERS)
    ordered_results: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _generate_single_file,
                step_file,
                thickness=thickness,
                finger_width=finger_width,
                kerf=kerf,
                layout=layout,
                sheet_width=sheet_width,
                sheet_height=sheet_height,
                part_gap=part_gap,
                sheet_gap=sheet_gap,
                job_dir=job_dir,
            ): step_file
            for step_file in deduped
        }

        for future in as_completed(futures):
            step_file = futures[future]
            try:
                ordered_results[step_file] = future.result()
            except Exception as exc:
                ordered_results[step_file] = {
                    "step_file": step_file,
                    "ok": False,
                    "error": str(exc),
                    "elapsed_s": 0.0,
                }

    results = [ordered_results[name] for name in deduped]

    file_entries: dict[str, dict[str, str]] = {}
    response_items: list[dict[str, Any]] = []

    for item in results:
        if not item.get("ok"):
            response_items.append(
                {
                    "step_file": item["step_file"],
                    "ok": False,
                    "error": item.get("error", "Unknown error"),
                    "elapsed_s": item.get("elapsed_s", 0.0),
                }
            )
            continue

        token = os.urandom(9).hex()
        file_entries[token] = {
            "path": item["output_path"],
            "filename": item["filename"],
            "step_file": item["step_file"],
        }

        response_items.append(
            {
                "step_file": item["step_file"],
                "ok": True,
                "filename": item["filename"],
                "elapsed_s": item.get("elapsed_s", 0.0),
                "download_url": f"/api/jobs/{job_id}/files/{token}",
                "preview_url": f"/api/jobs/{job_id}/files/{token}",
            }
        )

    zip_path: Path | None = None
    zip_url: str | None = None
    if file_entries:
        zip_path = job_dir / f"generated-{job_id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for info in file_entries.values():
                src = Path(info["path"])
                if src.exists():
                    zf.write(src, arcname=info["filename"])
        zip_url = f"/api/jobs/{job_id}/download.zip"

    with _JOB_LOCK:
        _JOB_INDEX[job_id] = {
            "created_at": time.time(),
            "job_dir": str(job_dir),
            "files": file_entries,
            "zip_path": str(zip_path) if zip_path else None,
        }

    success_count = sum(1 for item in response_items if item.get("ok"))
    return {
        "job_id": job_id,
        "layout": layout,
        "requested": len(deduped),
        "succeeded": success_count,
        "failed": len(deduped) - success_count,
        "zip_url": zip_url,
        "items": response_items,
    }


def _get_job(job_id: str) -> dict[str, Any]:
    _cleanup_expired_jobs()
    with _JOB_LOCK:
        job = _JOB_INDEX.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown or expired job")
        return job


def _render_index(error: str | None = None) -> str:
    options: list[str] = []
    for i, name in enumerate(_available_step_files()):
        escaped = html.escape(name)
        delay = f"animation-delay:{i * 0.04:.2f}s"
        options.append(
            "".join(
                [
                    f'<label class="step-choice" style="{delay}">',
                    f'<div class="step-thumb" data-step="{escaped}"><canvas></canvas></div>',
                    '<div class="step-info">',
                    f'<input type="checkbox" name="step_files" value="{escaped}" checked>',
                    f"<span>{escaped}</span>",
                    "</div>",
                    "</label>",
                ]
            )
        )

    if not options:
        options_html = '<p class="empty">No STEP files found in <code>step_files/</code>.</p>'
    else:
        options_html = "\n".join(options)

    error_html = ""
    if error:
        error_html = f'<p class="error">{html.escape(error)}</p>'

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>lasercut-gen</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f4f3ef;
      --surface: #ffffff;
      --surface-2: #f9f9f7;
      --border: #e6e5e1;
      --border-hi: #cdccc8;
      --text: #1a1a1a;
      --text-2: #6b6b6b;
      --text-3: #a0a09c;
      --accent: #ff6b2b;
      --accent-hover: #e85d20;
      --accent-dim: rgba(255,107,43,.07);
      --accent-border: rgba(255,107,43,.22);
      --danger: #e5484d;
      --danger-dim: rgba(229,72,77,.05);
      --mono: 'JetBrains Mono', 'SF Mono', monospace;
      --sans: 'Manrope', system-ui, sans-serif;
      --r: 10px;
      --r-sm: 6px;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 15px;
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
    }
    body::before {
      content: '';
      position: fixed; inset: 0;
      background-image: radial-gradient(circle, #cccbc7 0.6px, transparent 0.6px);
      background-size: 20px 20px;
      opacity: .45;
      mask-image: radial-gradient(ellipse 80% 50% at 50% 0%, black, transparent);
      -webkit-mask-image: radial-gradient(ellipse 80% 50% at 50% 0%, black, transparent);
      pointer-events: none; z-index: 0;
    }
    .wrap {
      position: relative; z-index: 1;
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px 20px 56px;
    }
    .header {
      display: flex; align-items: center; gap: 10px;
      margin-bottom: 24px;
    }
    .header h1 {
      font-family: var(--mono);
      font-size: 18px; font-weight: 700;
      color: var(--text);
      letter-spacing: -.02em;
    }
    .header .tag {
      font-family: var(--mono);
      font-size: 11px; font-weight: 500;
      color: var(--text-3);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 4px 12px;
      letter-spacing: .02em;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r);
      padding: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,.03), 0 6px 16px rgba(0,0,0,.03);
    }
    .panel + .panel { margin-top: 16px; }
    .hidden { display: none !important; }
    .section-label {
      font-family: var(--mono);
      font-size: 12px; font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .1em;
      color: var(--text-3);
      margin-bottom: 10px;
    }
    p.error {
      margin-bottom: 14px;
      color: var(--danger);
      font-family: var(--mono);
      font-size: 13px;
      background: var(--danger-dim);
      border: 1px solid rgba(229,72,77,.15);
      border-radius: var(--r-sm);
      padding: 10px 14px;
    }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }
    label.label {
      display: block;
      font-family: var(--mono);
      font-size: 12px; font-weight: 500;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--text-3);
      margin-bottom: 5px;
    }
    input[type="number"], select {
      width: 100%;
      border: 1.5px solid var(--border);
      border-radius: 8px;
      padding: 9px 11px;
      background: var(--surface);
      font-family: var(--mono);
      font-size: 14px;
      color: var(--text);
      outline: none;
      transition: border-color .15s, box-shadow .15s;
    }
    input[type="number"]:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }
    select { cursor: pointer; }
    .step-section {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }
    .step-tools {
      display: flex; gap: 6px; margin-bottom: 10px;
    }
    .tiny-btn {
      font-family: var(--mono);
      font-size: 12px; font-weight: 600;
      color: var(--text-2);
      background: var(--surface);
      border: 1.5px solid var(--border);
      border-radius: 6px;
      padding: 6px 14px;
      cursor: pointer;
      transition: all .15s;
    }
    .tiny-btn:hover { color: var(--text); border-color: var(--border-hi); background: #fff; }
    .step-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(185px, 1fr));
      gap: 8px;
    }
    @keyframes cardIn {
      from { opacity: 0; transform: translateY(5px) scale(.98); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    .step-choice {
      display: flex; flex-direction: column;
      border: 2px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      overflow: hidden; cursor: pointer;
      transition: border-color .2s, box-shadow .2s;
      animation: cardIn .3s ease both;
    }
    .step-choice:hover {
      border-color: var(--border-hi);
      box-shadow: 0 2px 8px rgba(0,0,0,.05);
    }
    .step-choice:has(input:checked) {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim), 0 2px 8px rgba(255,107,43,.08);
    }
    .step-thumb {
      height: 130px;
      background: var(--bg);
      position: relative;
      border-bottom: 1px solid var(--border);
    }
    .step-thumb::before {
      content: '';
      position: absolute; inset: 0;
      background-image: radial-gradient(circle, #c8c7c3 0.5px, transparent 0.5px);
      background-size: 12px 12px;
      opacity: .5;
      pointer-events: none;
    }
    .step-thumb canvas {
      position: relative; z-index: 1;
      width: 100%; height: 100%; display: block;
    }
    .step-info {
      display: flex; align-items: center; gap: 8px;
      padding: 9px 10px;
    }
    .step-info input[type="checkbox"] {
      appearance: none; -webkit-appearance: none;
      width: 16px; height: 16px; flex-shrink: 0;
      border: 2px solid var(--border-hi);
      border-radius: 4px;
      background: var(--surface);
      cursor: pointer; position: relative;
      transition: all .15s;
    }
    .step-info input[type="checkbox"]:checked {
      background: var(--accent);
      border-color: var(--accent);
    }
    .step-info input[type="checkbox"]:checked::after {
      content: '';
      position: absolute; left: 4px; top: 1.5px;
      width: 5px; height: 8px;
      border: solid #fff; border-width: 0 2px 2px 0;
      transform: rotate(45deg);
    }
    .step-choice span {
      font-family: var(--mono);
      font-size: 13px; color: var(--text-2);
      overflow: hidden; text-overflow: ellipsis;
      white-space: nowrap; min-width: 0;
      transition: color .15s;
    }
    .step-choice:has(input:checked) span { color: var(--text); font-weight: 500; }
    .empty { color: var(--text-3); font-family: var(--mono); font-size: 13px; }
    .packed-section {
      margin-top: 12px;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      display: none;
    }
    .packed-section.visible { display: block; }
    .packed-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }
    .cta-row {
      margin-top: 20px;
      display: flex; align-items: center; gap: 14px;
    }
    .cta {
      font-family: var(--mono);
      font-size: 14px; font-weight: 700;
      color: #fff;
      background: var(--accent);
      border: none;
      border-radius: 8px;
      padding: 12px 30px;
      cursor: pointer;
      transition: all .15s;
      letter-spacing: .01em;
      box-shadow: 0 2px 8px rgba(255,107,43,.2);
    }
    .cta:hover {
      background: var(--accent-hover);
      box-shadow: 0 4px 20px rgba(255,107,43,.28);
      transform: translateY(-1px);
    }
    .cta[disabled] { opacity: .5; cursor: wait; transform: none; box-shadow: none; }
    .status { font-family: var(--mono); font-size: 13px; color: var(--text-3); }
    .foot { margin-top: 16px; font-family: var(--mono); font-size: 12px; color: var(--text-3); }
    .results-head {
      display: flex; justify-content: space-between; align-items: center;
      gap: 12px; flex-wrap: wrap; margin-bottom: 10px;
    }
    .results-head h2 {
      font-family: var(--mono);
      font-size: 16px; font-weight: 700;
      color: var(--text);
    }
    .download-zip {
      font-family: var(--mono);
      font-size: 13px; font-weight: 600;
      color: var(--accent);
      background: var(--accent-dim);
      border: 1px solid var(--accent-border);
      border-radius: var(--r-sm);
      padding: 7px 14px;
      text-decoration: none;
      transition: all .15s;
    }
    .download-zip:hover { background: rgba(255,107,43,.12); border-color: var(--accent); }
    .results-meta {
      font-family: var(--mono);
      font-size: 13px; color: var(--text-3);
      margin-bottom: 12px;
    }
    .results-panel {
      position: relative; z-index: 1;
      margin: 16px 20px 56px;
      border-radius: var(--r);
    }
    .results-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .result-card {
      border: 1px solid var(--border);
      border-radius: var(--r);
      background: var(--surface);
      padding: 16px; display: grid; gap: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,.02);
    }
    .result-card.error {
      border-color: rgba(229,72,77,.2);
      background: var(--danger-dim);
    }
    .result-head {
      display: flex; justify-content: space-between;
      gap: 10px; align-items: center; flex-wrap: wrap;
    }
    .result-title {
      font-family: var(--mono);
      font-size: 14px; font-weight: 600;
      color: var(--text); overflow-wrap: anywhere;
    }
    .result-meta { font-family: var(--mono); font-size: 12px; color: var(--text-3); }
    .download-one {
      font-family: var(--mono);
      font-size: 12px; font-weight: 600;
      color: var(--accent);
      border: 1px solid var(--accent-border);
      background: var(--accent-dim);
      border-radius: var(--r-sm);
      padding: 5px 12px;
      text-decoration: none;
      transition: all .15s;
    }
    .download-one:hover { border-color: var(--accent); background: rgba(255,107,43,.12); }
    .svg-wrap {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .svg-inner { width: 100%; }
    .svg-inner img {
      display: block; width: 100%; height: auto;
      filter: drop-shadow(0 0 0.6px #000) drop-shadow(0 0 0.6px #000) drop-shadow(0 0 0.6px #000);
    }
    .svg-tools {
      display: flex; justify-content: flex-end;
      align-items: center; gap: 8px;
    }
    .svg-open {
      font-family: var(--mono);
      font-size: 12px; color: var(--text-3);
      text-decoration: none; transition: color .15s;
    }
    .svg-open:hover { color: var(--text-2); }
    .svg-modal-btn {
      font-family: var(--mono);
      font-size: 12px; font-weight: 600;
      color: var(--text-2);
      background: var(--surface-2);
      border: 1.5px solid var(--border);
      border-radius: var(--r-sm);
      padding: 5px 14px;
      cursor: pointer; transition: all .15s;
    }
    .svg-modal-btn:hover { color: var(--text); border-color: var(--border-hi); }
    .modal-overlay {
      position: fixed; inset: 0;
      background: rgba(244,243,239,.9);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      z-index: 1000;
      display: flex; align-items: center; justify-content: center;
      padding: 28px;
      animation: fadeIn .15s ease;
    }
    .modal-overlay.hidden { display: none; }
    @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
    .modal-content {
      position: relative;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r);
      max-width: 96vw; max-height: 94vh;
      overflow: auto;
      box-shadow: 0 24px 80px rgba(0,0,0,.12), 0 2px 6px rgba(0,0,0,.06);
    }
    .modal-close {
      position: fixed; top: 16px; right: 20px;
      width: 36px; height: 36px;
      display: flex; align-items: center; justify-content: center;
      background: var(--surface);
      border: 1.5px solid var(--border);
      border-radius: 8px;
      font-size: 16px; cursor: pointer;
      color: var(--text-2); z-index: 1001;
      transition: all .15s;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }
    .modal-close:hover { color: var(--text); border-color: var(--border-hi); }
    .modal-content img { display: block; width: 100%; height: auto; }
    .failure {
      font-family: var(--mono);
      color: var(--danger);
      font-size: 13px; line-height: 1.5;
      white-space: pre-wrap;
    }
    .result-card.loading { border-color: var(--border); }
    .loading-skeleton {
      height: 120px;
      border-radius: 8px;
      background: linear-gradient(90deg, var(--surface-2) 25%, #eeeee8 50%, var(--surface-2) 75%);
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
    }
    @keyframes shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }
    .result-card.loading .result-meta {
      color: var(--accent);
    }
    @media (max-width: 900px) {
      .form-grid { grid-template-columns: 1fr 1fr; }
      .packed-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 600px) {
      .form-grid, .packed-grid { grid-template-columns: 1fr; }
      .step-grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>// lasercut-gen</h1>
      <span class="tag">STEP &rarr; SVG</span>
    </div>

    <section class="panel">
      __ERROR_BLOCK__
      <form id="generate-form">
        <div class="section-label">Models</div>
        <div class="step-section">
          <div class="step-tools">
            <button type="button" class="tiny-btn" id="select-all">All</button>
            <button type="button" class="tiny-btn" id="select-none">None</button>
          </div>
          <div class="step-grid" id="step-grid">__STEP_OPTIONS__</div>
        </div>

        <div style="margin-top:18px">
          <div class="section-label">Parameters</div>
          <div class="form-grid">
            <div>
              <label class="label" for="layout">Layout</label>
              <select id="layout" name="layout">
                <option value="packed" selected>packed</option>
                <option value="unfolded">unfolded</option>
              </select>
            </div>
            <div>
              <label class="label" for="finger_width">Finger Width (mm)</label>
              <input id="finger_width" name="finger_width" type="number" min="1" step="0.1" value="20.0">
            </div>
            <div>
              <label class="label" for="thickness">Thickness (mm)</label>
              <input id="thickness" name="thickness" type="number" min="0.1" step="0.01" value="3">
            </div>
            <div>
              <label class="label" for="kerf">Kerf (mm)</label>
              <input id="kerf" name="kerf" type="number" step="0.01" value="0.02">
            </div>
          </div>
        </div>

        <div id="packed-block" class="packed-section">
          <div class="section-label">Sheet</div>
          <div class="packed-grid">
            <div>
              <label class="label" for="sheet_width">Width (mm)</label>
              <input id="sheet_width" name="sheet_width" type="number" min="1" step="1" value="600">
            </div>
            <div>
              <label class="label" for="sheet_height">Height (mm)</label>
              <input id="sheet_height" name="sheet_height" type="number" min="1" step="1" value="400">
            </div>
            <div>
              <label class="label" for="part_gap">Part Gap (mm)</label>
              <input id="part_gap" name="part_gap" type="number" min="0" step="0.1" value="4">
            </div>
            <div>
              <label class="label" for="sheet_gap">Sheet Gap (mm)</label>
              <input id="sheet_gap" name="sheet_gap" type="number" min="0" step="0.1" value="20">
            </div>
          </div>
        </div>

        <div class="cta-row">
          <button id="generate-btn" class="cta" type="submit">Generate</button>
          <span id="status" class="status">Bereit</span>
        </div>
      </form>
      <p class="foot">Jobs are stored temporarily and removed automatically after a few hours.</p>
    </section>

  </div>

  <section id="results-panel" class="panel results-panel hidden">
    <div class="results-head">
      <h2>// results</h2>
      <div id="zip-slot"></div>
    </div>
    <p id="results-meta" class="results-meta"></p>
    <div id="results-grid" class="results-grid"></div>
  </section>

  <div id="svg-modal" class="modal-overlay hidden">
    <button class="modal-close" id="modal-close-btn">&#x2715;</button>
    <div class="modal-content">
      <img id="svg-modal-img" src="" alt="SVG Preview">
    </div>
  </div>

  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
  <script>
    const form = document.getElementById('generate-form');
    const layoutSel = document.getElementById('layout');
    const packedBlock = document.getElementById('packed-block');
    const statusEl = document.getElementById('status');
    const generateBtn = document.getElementById('generate-btn');
    const resultsPanel = document.getElementById('results-panel');
    const resultsMeta = document.getElementById('results-meta');
    const resultsGrid = document.getElementById('results-grid');
    const zipSlot = document.getElementById('zip-slot');
    const selectAllBtn = document.getElementById('select-all');
    const selectNoneBtn = document.getElementById('select-none');
    const svgModal = document.getElementById('svg-modal');
    const svgModalImg = document.getElementById('svg-modal-img');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const stepPreviewCache = new Map();
    const stepCheckboxes = Array.from(document.querySelectorAll('input[name="step_files"]'));
    const SETTINGS_KEY = 'lasercut.web.settings.v1';
    const fingerWidthInput = document.getElementById('finger_width');
    const thicknessInput = document.getElementById('thickness');
    const kerfInput = document.getElementById('kerf');
    const sheetWidthInput = document.getElementById('sheet_width');
    const sheetHeightInput = document.getElementById('sheet_height');
    const partGapInput = document.getElementById('part_gap');
    const sheetGapInput = document.getElementById('sheet_gap');
    const persistedInputs = [
      fingerWidthInput,
      thicknessInput,
      kerfInput,
      sheetWidthInput,
      sheetHeightInput,
      partGapInput,
      sheetGapInput,
    ];

    function syncLayout() {
      packedBlock.classList.toggle('visible', layoutSel.value === 'packed');
    }

    function selectedFiles() {
      return Array.from(document.querySelectorAll('input[name="step_files"]:checked'))
        .map((cb) => cb.value);
    }

    function setAllStepCheckboxes(checked) {
      document.querySelectorAll('input[name="step_files"]').forEach((cb) => {
        cb.checked = checked;
      });
    }

    function _readStoredSettings() {
      try {
        const raw = window.localStorage.getItem(SETTINGS_KEY);
        if (!raw) {
          return null;
        }
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' ? parsed : null;
      } catch (_) {
        return null;
      }
    }

    function saveSettings() {
      const payload = {
        layout: layoutSel.value,
        finger_width: fingerWidthInput ? fingerWidthInput.value : '',
        thickness: thicknessInput ? thicknessInput.value : '',
        kerf: kerfInput ? kerfInput.value : '',
        sheet_width: sheetWidthInput ? sheetWidthInput.value : '',
        sheet_height: sheetHeightInput ? sheetHeightInput.value : '',
        part_gap: partGapInput ? partGapInput.value : '',
        sheet_gap: sheetGapInput ? sheetGapInput.value : '',
        step_files: selectedFiles(),
      };
      try {
        window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(payload));
      } catch (_) {}
    }

    function restoreSettings() {
      const stored = _readStoredSettings();
      if (!stored) {
        return;
      }

      const applyTextValue = (el, key) => {
        if (!el) {
          return;
        }
        const value = stored[key];
        if (typeof value === 'string') {
          el.value = value;
        }
      };

      applyTextValue(layoutSel, 'layout');
      applyTextValue(fingerWidthInput, 'finger_width');
      applyTextValue(thicknessInput, 'thickness');
      applyTextValue(kerfInput, 'kerf');
      applyTextValue(sheetWidthInput, 'sheet_width');
      applyTextValue(sheetHeightInput, 'sheet_height');
      applyTextValue(partGapInput, 'part_gap');
      applyTextValue(sheetGapInput, 'sheet_gap');

      if (Array.isArray(stored.step_files)) {
        const wanted = new Set(stored.step_files.filter((item) => typeof item === 'string'));
        stepCheckboxes.forEach((cb) => {
          cb.checked = wanted.has(cb.value);
        });
      }
    }

    function clearViewer(canvas) {
      const cleanup = canvas.__viewerCleanup;
      if (typeof cleanup === 'function') {
        cleanup();
      }
      canvas.__viewerCleanup = null;
    }

    function drawMeshUnavailable(canvas, message) {
      clearViewer(canvas);
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const w = canvas.clientWidth || 400;
      const h = canvas.clientHeight || 130;
      canvas.width = w; canvas.height = h;
      ctx.fillStyle = '#f4f3ef';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#a0a09c';
      ctx.font = '13px JetBrains Mono, monospace';
      ctx.fillText(message, 12, 22);
    }

    function initFixedMeshViewer(canvas, meshData) {
      if (!window.THREE || !meshData || !meshData.vertices || !meshData.triangles || meshData.triangles.length === 0) {
        drawMeshUnavailable(canvas, 'no preview');
        return;
      }

      clearViewer(canvas);
      const THREE = window.THREE;
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf4f3ef);
      const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 20000);

      const triCount = meshData.triangles.length;
      const positions = new Float32Array(triCount * 9);
      let ptr = 0;

      for (const tri of meshData.triangles) {
        const a = meshData.vertices[tri[0]];
        const b = meshData.vertices[tri[1]];
        const c = meshData.vertices[tri[2]];
        positions[ptr++] = a[0]; positions[ptr++] = a[1]; positions[ptr++] = a[2];
        positions[ptr++] = b[0]; positions[ptr++] = b[1]; positions[ptr++] = b[2];
        positions[ptr++] = c[0]; positions[ptr++] = c[1]; positions[ptr++] = c[2];
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.computeVertexNormals();

      const group = new THREE.Group();
      const solidMaterial = new THREE.MeshStandardMaterial({
        color: 0xd8d4cc,
        metalness: 0.05,
        roughness: 0.8,
        side: THREE.DoubleSide,
      });
      const edgeGeometry = new THREE.EdgesGeometry(geometry, 22);
      const edgeMaterial = new THREE.LineBasicMaterial({ color: 0xff6b2b });
      const solid = new THREE.Mesh(geometry, solidMaterial);
      const wire = new THREE.LineSegments(edgeGeometry, edgeMaterial);
      group.add(solid);
      group.add(wire);
      // CadQuery uses Z-up, Three.js uses Y-up
      group.rotation.x = -Math.PI / 2;
      scene.add(group);

      const bbox = new THREE.Box3().setFromObject(group);
      const center = bbox.getCenter(new THREE.Vector3());
      group.position.set(-center.x, -center.y, -center.z);

      const size = bbox.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z) * 0.55 || 1;
      camera.position.set(radius * 2.0, radius * 1.4, radius * 2.0);
      camera.lookAt(0, 0, 0);

      scene.add(new THREE.AmbientLight(0xffffff, 0.5));
      const key = new THREE.DirectionalLight(0xffffff, 0.7);
      key.position.set(2, 3, 1.5);
      scene.add(key);
      const rim = new THREE.DirectionalLight(0xffeedd, 0.3);
      rim.position.set(-1.5, -1, -2);
      scene.add(rim);

      const render = () => {
        renderer.render(scene, camera);
      };
      const resize = () => {
        const w = Math.max(10, canvas.clientWidth);
        const h = Math.max(10, canvas.clientHeight);
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        render();
      };

      const observer = new ResizeObserver(resize);
      observer.observe(canvas);
      resize();

      canvas.__viewerCleanup = () => {
        observer.disconnect();
        renderer.dispose();
        edgeGeometry.dispose();
        geometry.dispose();
        edgeMaterial.dispose();
        solidMaterial.dispose();
      };
    }

    async function parseJsonResponse(resp) {
      let payload = null;
      try {
        payload = await resp.json();
      } catch (_) {
        payload = null;
      }

      if (!resp.ok) {
        const detail = payload && payload.detail ? payload.detail : `HTTP ${resp.status}`;
        throw new Error(detail);
      }

      return payload;
    }

    async function fetchStepPreviewMesh(stepFile) {
      if (stepPreviewCache.has(stepFile)) {
        return stepPreviewCache.get(stepFile);
      }
      const encoded = encodeURIComponent(stepFile);
      const resp = await fetch(`/api/step-preview/${encoded}`);
      const payload = await parseJsonResponse(resp);
      const mesh = payload && payload.mesh ? payload.mesh : null;
      stepPreviewCache.set(stepFile, mesh);
      return mesh;
    }

    async function initStepThumbnails() {
      const thumbs = document.querySelectorAll('.step-thumb[data-step]');
      const promises = Array.from(thumbs).map(async (thumb) => {
        const stepFile = thumb.getAttribute('data-step');
        const canvas = thumb.querySelector('canvas');
        if (!stepFile || !canvas) return;
        try {
          const mesh = await fetchStepPreviewMesh(stepFile);
          initFixedMeshViewer(canvas, mesh);
        } catch (_) {
          drawMeshUnavailable(canvas, 'preview unavailable');
        }
      });
      await Promise.all(promises);
    }

    function openSvgModal(url) {
      svgModalImg.src = url;
      svgModal.classList.remove('hidden');
      document.body.style.overflow = 'hidden';
    }

    function closeSvgModal() {
      svgModal.classList.add('hidden');
      svgModalImg.src = '';
      document.body.style.overflow = '';
    }

    function buildErrorCard(item) {
      const card = document.createElement('article');
      card.className = 'result-card error';

      const head = document.createElement('div');
      head.className = 'result-head';
      const title = document.createElement('div');
      title.className = 'result-title';
      title.textContent = item.step_file;
      const meta = document.createElement('div');
      meta.className = 'result-meta';
      meta.textContent = `failed in ${Number(item.elapsed_s || 0).toFixed(2)}s`;
      head.appendChild(title);
      head.appendChild(meta);

      const err = document.createElement('div');
      err.className = 'failure';
      err.textContent = item.error || 'Unknown error';

      card.appendChild(head);
      card.appendChild(err);
      return card;
    }

    function buildSuccessCard(item) {
      const card = document.createElement('article');
      card.className = 'result-card';

      const head = document.createElement('div');
      head.className = 'result-head';

      const title = document.createElement('div');
      title.className = 'result-title';
      title.textContent = item.step_file;

      const right = document.createElement('div');
      right.style.display = 'flex';
      right.style.gap = '8px';
      right.style.alignItems = 'center';

      const meta = document.createElement('div');
      meta.className = 'result-meta';
      meta.textContent = `${item.filename} · ${Number(item.elapsed_s || 0).toFixed(2)}s`;

      const dl = document.createElement('a');
      dl.className = 'download-one';
      dl.href = item.download_url;
      dl.download = item.filename;
      dl.textContent = 'Download';

      right.appendChild(meta);
      right.appendChild(dl);
      head.appendChild(title);
      head.appendChild(right);

      const svgWrap = document.createElement('div');
      svgWrap.className = 'svg-wrap';
      const svgInner = document.createElement('div');
      svgInner.className = 'svg-inner';
      const svgImg = document.createElement('img');
      svgImg.src = item.preview_url;
      svgImg.alt = `${item.step_file} preview`;
      svgImg.loading = 'lazy';
      svgInner.appendChild(svgImg);
      svgWrap.appendChild(svgInner);

      const svgTools = document.createElement('div');
      svgTools.className = 'svg-tools';

      const modalBtn = document.createElement('button');
      modalBtn.className = 'svg-modal-btn';
      modalBtn.textContent = 'Enlarge';
      modalBtn.addEventListener('click', () => openSvgModal(item.preview_url));

      const open = document.createElement('a');
      open.className = 'svg-open';
      open.href = item.preview_url;
      open.target = '_blank';
      open.rel = 'noreferrer';
      open.textContent = 'Open SVG';

      svgTools.appendChild(modalBtn);
      svgTools.appendChild(open);

      card.appendChild(head);
      card.appendChild(svgWrap);
      card.appendChild(svgTools);

      return card;
    }

    function renderResults(payload) {
      const items = payload.items || [];
      resultsGrid.innerHTML = '';
      zipSlot.innerHTML = '';

      if (payload.zip_url) {
        const zip = document.createElement('a');
        zip.className = 'download-zip';
        zip.href = payload.zip_url;
        zip.textContent = 'Download ZIP';
        zipSlot.appendChild(zip);
      }

      resultsMeta.textContent = `${payload.succeeded}/${payload.requested} succeeded · ${payload.failed} failed · layout=${payload.layout}`;
      resultsPanel.classList.remove('hidden');

      items.forEach((item) => {
        const card = item.ok ? buildSuccessCard(item) : buildErrorCard(item);
        resultsGrid.appendChild(card);
      });
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();

      const chosen = selectedFiles();
      if (chosen.length === 0) {
        statusEl.textContent = 'Please select at least one box.';
        return;
      }

      generateBtn.disabled = true;
      while (resultsGrid.firstChild) resultsGrid.removeChild(resultsGrid.firstChild);
      while (zipSlot.firstChild) zipSlot.removeChild(zipSlot.firstChild);
      resultsMeta.textContent = '';
      resultsPanel.classList.remove('hidden');

      let completed = 0;
      const total = chosen.length;
      const fileResults = [];
      const cardMap = {};

      statusEl.textContent = `Generating 0/${total} ...`;

      chosen.forEach((file) => {
        const card = document.createElement('article');
        card.className = 'result-card loading';
        const head = document.createElement('div');
        head.className = 'result-head';
        const title = document.createElement('div');
        title.className = 'result-title';
        title.textContent = file;
        const meta = document.createElement('div');
        meta.className = 'result-meta';
        meta.textContent = 'generating\u2026';
        head.appendChild(title);
        head.appendChild(meta);
        const skeleton = document.createElement('div');
        skeleton.className = 'loading-skeleton';
        card.appendChild(head);
        card.appendChild(skeleton);
        resultsGrid.appendChild(card);
        cardMap[file] = card;
      });

      const promises = chosen.map(async (file) => {
        const fd = new FormData();
        fd.set('step_file', file);
        fd.set('thickness', thicknessInput.value);
        fd.set('finger_width', fingerWidthInput.value);
        fd.set('kerf', kerfInput.value);
        fd.set('layout', layoutSel.value);
        fd.set('sheet_width', sheetWidthInput.value);
        fd.set('sheet_height', sheetHeightInput.value);
        fd.set('part_gap', partGapInput.value);
        fd.set('sheet_gap', sheetGapInput.value);

        try {
          const resp = await fetch('/api/generate-single', {
            method: 'POST',
            body: fd,
          });
          const result = await parseJsonResponse(resp);
          fileResults.push(result);
          const placeholder = cardMap[file];
          if (placeholder) {
            const newCard = result.ok ? buildSuccessCard(result) : buildErrorCard(result);
            newCard.style.animation = 'cardIn .3s ease both';
            placeholder.replaceWith(newCard);
          }
        } catch (error) {
          const errResult = {
            step_file: file,
            ok: false,
            error: error instanceof Error ? error.message : String(error),
            elapsed_s: 0,
          };
          fileResults.push(errResult);
          const placeholder = cardMap[file];
          if (placeholder) {
            placeholder.replaceWith(buildErrorCard(errResult));
          }
        }

        completed++;
        statusEl.textContent = `Generated ${completed}/${total} ...`;
      });

      await Promise.all(promises);

      const succeeded = fileResults.filter((r) => r.ok).length;
      const failed = total - succeeded;
      statusEl.textContent = `Done: ${succeeded}/${total} succeeded.`;
      resultsMeta.textContent = `${succeeded}/${total} succeeded \u00b7 ${failed} failed \u00b7 layout=${layoutSel.value}`;

      const successResults = fileResults.filter((r) => r.ok && r.download_url);
      if (successResults.length > 0) {
        try {
          const refs = successResults.map((r) => {
            const parts = r.download_url.split('/');
            return { job_id: parts[3], file_token: parts[5] };
          });
          const zipResp = await fetch('/api/bundle-zip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: refs }),
          });
          const zipData = await parseJsonResponse(zipResp);
          if (zipData.zip_url) {
            const zip = document.createElement('a');
            zip.className = 'download-zip';
            zip.href = zipData.zip_url;
            zip.textContent = 'Download ZIP';
            zipSlot.appendChild(zip);
          }
        } catch (_) {
          /* zip bundling failed; individual downloads still work */
        }
      }

      generateBtn.disabled = false;
    });

    selectAllBtn.addEventListener('click', () => {
      setAllStepCheckboxes(true);
      saveSettings();
    });
    selectNoneBtn.addEventListener('click', () => {
      setAllStepCheckboxes(false);
      saveSettings();
    });
    stepCheckboxes.forEach((cb) => {
      cb.addEventListener('change', saveSettings);
    });
    persistedInputs.forEach((el) => {
      if (!el) {
        return;
      }
      el.addEventListener('change', saveSettings);
      el.addEventListener('input', saveSettings);
    });
    layoutSel.addEventListener('change', () => {
      syncLayout();
      saveSettings();
    });
    modalCloseBtn.addEventListener('click', closeSvgModal);
    svgModal.addEventListener('click', (e) => {
      if (e.target === svgModal) closeSvgModal();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeSvgModal();
    });
    restoreSettings();
    syncLayout();
    initStepThumbnails();
  </script>
</body>
</html>"""

    return template.replace("__STEP_OPTIONS__", options_html).replace("__ERROR_BLOCK__", error_html)


app = FastAPI(title="Lego Sorter Bin Generator", version="0.2.0")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_render_index())


@app.get("/healthz", response_class=JSONResponse)
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/camera-test", response_class=HTMLResponse)
def camera_test() -> HTMLResponse:
    # Try project root (dev) then /app (Docker)
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "camera_test.html",
        Path("/app/camera_test.html"),
    ]
    for p in candidates:
        if p.exists():
            return HTMLResponse(p.read_text())
    raise HTTPException(status_code=404, detail="camera_test.html not found")


@app.get("/api/step-preview/{step_file:path}", response_class=JSONResponse)
def step_preview(step_file: str) -> JSONResponse:
    available = _available_step_files()
    if step_file not in available:
        raise HTTPException(status_code=404, detail=f"Unknown step file: {step_file}")

    mesh = _get_step_preview_mesh(step_file)
    return JSONResponse(
        {
            "step_file": step_file,
            "mesh": mesh,
        }
    )


@app.post("/api/generate-batch", response_class=JSONResponse)
def generate_batch(
    step_files: list[str] = Form(...),
    thickness: float = Form(3.2),
    finger_width: float = Form(20.0),
    kerf: float = Form(0.0),
    layout: str = Form("unfolded"),
    sheet_width: str | None = Form(None),
    sheet_height: str | None = Form(None),
    part_gap: float = Form(4.0),
    sheet_gap: float = Form(20.0),
) -> JSONResponse:
    if layout not in {"unfolded", "packed"}:
        raise HTTPException(status_code=400, detail="layout must be 'unfolded' or 'packed'")

    try:
        parsed_sheet_width = _parse_optional_float(sheet_width)
        parsed_sheet_height = _parse_optional_float(sheet_height)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sheet size: {exc}") from exc

    if layout == "packed":
        if parsed_sheet_width is None or parsed_sheet_height is None:
            raise HTTPException(status_code=400, detail="packed layout requires sheet width and height")
        if parsed_sheet_width <= 0 or parsed_sheet_height <= 0:
            raise HTTPException(status_code=400, detail="sheet dimensions must be > 0")

    payload = _run_batch_generation(
        step_files,
        thickness=thickness,
        finger_width=finger_width,
        kerf=kerf,
        layout=layout,
        sheet_width=parsed_sheet_width,
        sheet_height=parsed_sheet_height,
        part_gap=part_gap,
        sheet_gap=sheet_gap,
    )
    return JSONResponse(payload)


@app.post("/api/generate-single", response_class=JSONResponse)
def generate_single_api(
    step_file: str = Form(...),
    thickness: float = Form(3.2),
    finger_width: float = Form(20.0),
    kerf: float = Form(0.0),
    layout: str = Form("unfolded"),
    sheet_width: str | None = Form(None),
    sheet_height: str | None = Form(None),
    part_gap: float = Form(4.0),
    sheet_gap: float = Form(20.0),
) -> JSONResponse:
    if layout not in {"unfolded", "packed"}:
        raise HTTPException(status_code=400, detail="layout must be 'unfolded' or 'packed'")

    try:
        parsed_sheet_width = _parse_optional_float(sheet_width)
        parsed_sheet_height = _parse_optional_float(sheet_height)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sheet size: {exc}") from exc

    if layout == "packed":
        if parsed_sheet_width is None or parsed_sheet_height is None:
            raise HTTPException(status_code=400, detail="packed layout requires sheet width and height")
        if parsed_sheet_width <= 0 or parsed_sheet_height <= 0:
            raise HTTPException(status_code=400, detail="sheet dimensions must be > 0")

    available = set(_available_step_files())
    if step_file not in available:
        raise HTTPException(status_code=404, detail=f"Unknown step file: {step_file}")

    _cleanup_expired_jobs()

    job_id = f"{int(time.time())}-{os.urandom(4).hex()}"
    job_dir = _job_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    result = _generate_single_file(
        step_file,
        thickness=thickness,
        finger_width=finger_width,
        kerf=kerf,
        layout=layout,
        sheet_width=parsed_sheet_width,
        sheet_height=parsed_sheet_height,
        part_gap=part_gap,
        sheet_gap=sheet_gap,
        job_dir=job_dir,
    )

    file_entries: dict[str, dict[str, str]] = {}
    response: dict[str, Any] = {
        "job_id": job_id,
        "step_file": result["step_file"],
        "ok": result.get("ok", False),
        "elapsed_s": result.get("elapsed_s", 0.0),
    }

    if result.get("ok"):
        token = os.urandom(9).hex()
        file_entries[token] = {
            "path": result["output_path"],
            "filename": result["filename"],
            "step_file": result["step_file"],
        }
        response["filename"] = result["filename"]
        response["download_url"] = f"/api/jobs/{job_id}/files/{token}"
        response["preview_url"] = f"/api/jobs/{job_id}/files/{token}"
    else:
        response["error"] = result.get("error", "Unknown error")

    with _JOB_LOCK:
        _JOB_INDEX[job_id] = {
            "created_at": time.time(),
            "job_dir": str(job_dir),
            "files": file_entries,
            "zip_path": None,
        }

    return JSONResponse(response)


@app.get("/api/jobs/{job_id}/files/{file_token}")
def download_job_file(job_id: str, file_token: str) -> FileResponse:
    job = _get_job(job_id)
    files = job.get("files", {})
    info = files.get(file_token)
    if info is None:
        raise HTTPException(status_code=404, detail="Unknown file token")

    path = Path(info["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "image/svg+xml" if path.suffix.lower() == ".svg" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=info["filename"])


@app.get("/api/jobs/{job_id}/download.zip")
def download_job_zip(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    zip_path_raw = job.get("zip_path")
    if not zip_path_raw:
        raise HTTPException(status_code=404, detail="ZIP not available for this job")

    zip_path = Path(zip_path_raw)
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="ZIP file not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"lasercut-{job_id}.zip",
    )


@app.post("/api/bundle-zip", response_class=JSONResponse)
async def bundle_zip(request: Request) -> JSONResponse:
    body = await request.json()
    file_refs = body.get("files", [])

    if not file_refs:
        raise HTTPException(status_code=400, detail="No files to bundle")

    candidates: list[dict[str, str]] = []
    with _JOB_LOCK:
        for ref in file_refs:
            jid = ref.get("job_id", "")
            tok = ref.get("file_token", "")
            job = _JOB_INDEX.get(jid)
            if not job:
                continue
            info = job.get("files", {}).get(tok)
            if info:
                candidates.append({"path": info["path"], "filename": info["filename"]})

    entries = [c for c in candidates if Path(c["path"]).exists()]

    if not entries:
        raise HTTPException(status_code=400, detail="No valid files found")

    zip_job_id = f"{int(time.time())}-{os.urandom(4).hex()}"
    zip_job_dir = _job_root() / zip_job_id
    zip_job_dir.mkdir(parents=True, exist_ok=True)

    zip_path = zip_job_dir / f"generated-{zip_job_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            zf.write(entry["path"], arcname=entry["filename"])

    with _JOB_LOCK:
        _JOB_INDEX[zip_job_id] = {
            "created_at": time.time(),
            "job_dir": str(zip_job_dir),
            "files": {},
            "zip_path": str(zip_path),
        }

    return JSONResponse({"zip_url": f"/api/jobs/{zip_job_id}/download.zip"})


@app.post("/generate")
def generate_single_legacy(
    step_file: str = Form(...),
    thickness: float = Form(3.2),
    finger_width: float = Form(20.0),
    kerf: float = Form(0.0),
    layout: str = Form("unfolded"),
    sheet_width: str | None = Form(None),
    sheet_height: str | None = Form(None),
    part_gap: float = Form(4.0),
    sheet_gap: float = Form(20.0),
) -> FileResponse:
    if layout not in {"unfolded", "packed"}:
        raise HTTPException(status_code=400, detail="layout must be 'unfolded' or 'packed'")

    try:
        parsed_sheet_width = _parse_optional_float(sheet_width)
        parsed_sheet_height = _parse_optional_float(sheet_height)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sheet size: {exc}") from exc

    if layout == "packed":
        if parsed_sheet_width is None or parsed_sheet_height is None:
            raise HTTPException(status_code=400, detail="packed layout requires sheet width and height")
        if parsed_sheet_width <= 0 or parsed_sheet_height <= 0:
            raise HTTPException(status_code=400, detail="sheet dimensions must be > 0")

    payload = _run_batch_generation(
        [step_file],
        thickness=thickness,
        finger_width=finger_width,
        kerf=kerf,
        layout=layout,
        sheet_width=parsed_sheet_width,
        sheet_height=parsed_sheet_height,
        part_gap=part_gap,
        sheet_gap=sheet_gap,
    )

    items = payload.get("items", [])
    if not items:
        raise HTTPException(status_code=500, detail="No output generated")

    first = items[0]
    if not first.get("ok"):
        raise HTTPException(status_code=500, detail=first.get("error", "Generation failed"))

    dl = first.get("download_url")
    if not isinstance(dl, str):
        raise HTTPException(status_code=500, detail="Missing output URL")

    file_token = dl.rsplit("/", 1)[-1]
    return download_job_file(payload["job_id"], file_token)
