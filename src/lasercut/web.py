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
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from lasercut.exporter import export_svg
from lasercut.joints import apply_finger_joints
from lasercut.panels import load_step_panels


_JOB_LOCK = Lock()
_JOB_INDEX: dict[str, dict[str, Any]] = {}
_JOB_TTL_SECONDS = int(float(os.getenv("LASERCUT_WEB_JOB_TTL_SECONDS", "21600")))
_JOB_MAX_WORKERS = max(1, int(float(os.getenv("LASERCUT_WEB_MAX_WORKERS", "4"))))
_FIXED_PACK_ROTATIONS = 8


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

        mesh = _mesh_from_step(step_path)

        elapsed = round(time.perf_counter() - started, 2)
        return {
            "step_file": step_file,
            "ok": True,
            "filename": filename,
            "folder_name": folder_name,
            "output_path": str(output_path),
            "mesh": mesh,
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
                "mesh": item.get("mesh"),
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
    for name in _available_step_files():
        escaped = html.escape(name)
        options.append(
            "".join(
                [
                    '<label class="step-choice">',
                    f'<input type="checkbox" name="step_files" value="{escaped}" checked>',
                    f"<span>{escaped}</span>",
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
  <title>Lego Sorter Bin Generator</title>
  <style>
    :root {
      --bg: #f3f5f1;
      --paper: #ffffff;
      --ink: #162019;
      --muted: #57645b;
      --line: #cfd7d1;
      --accent: #22613e;
      --accent-soft: #e8f0ea;
      --danger: #8a1d1d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(1200px 600px at 0% -10%, #e8efe8 0%, var(--bg) 55%, #eef2ee 100%);
      color: var(--ink);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    .wrap {
      max-width: 1280px;
      margin: 22px auto 36px;
      padding: 0 16px;
    }
    .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 14px 34px rgba(20, 30, 20, 0.08);
    }
    .hidden { display: none; }
    h1 {
      margin: 0 0 6px;
      font-size: 27px;
      letter-spacing: .2px;
    }
    .sub {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 14px;
    }
    .error {
      margin: 0 0 10px;
      color: var(--danger);
      font-weight: 700;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .full { grid-column: 1 / -1; }
    label.label {
      display: block;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .05em;
      color: #2b352d;
      margin-bottom: 6px;
    }
    input[type="number"], select {
      width: 100%;
      border: 1px solid #c4cec6;
      border-radius: 9px;
      padding: 9px 10px;
      background: #fff;
      font-size: 14px;
      color: var(--ink);
    }
    .step-box {
      border: 1px dashed #c8d2ca;
      border-radius: 12px;
      background: #fbfcfa;
      padding: 10px;
    }
    .step-tools {
      display: flex;
      gap: 8px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }
    .tiny-btn {
      border: 1px solid #bcc8bf;
      background: #fff;
      color: #213128;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 700;
      padding: 6px 9px;
      cursor: pointer;
    }
    .tiny-btn:hover { background: #f5f8f5; }
    .step-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 8px;
      max-height: 230px;
      overflow: auto;
      padding-right: 4px;
    }
    .step-choice {
      display: flex;
      align-items: center;
      gap: 8px;
      border: 1px solid #d6ddd7;
      border-radius: 9px;
      padding: 7px 8px;
      background: #fff;
      min-height: 36px;
    }
    .step-choice span {
      font-size: 13px;
      color: #243129;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .empty {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }
    .packed {
      border: 1px dashed #ccd5ce;
      border-radius: 11px;
      padding: 10px;
      background: #fafcf9;
      display: none;
    }
    .packed.visible { display: block; }
    .cta-row {
      margin-top: 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .cta {
      border: none;
      border-radius: 10px;
      padding: 10px 16px;
      font-size: 14px;
      font-weight: 800;
      letter-spacing: .02em;
      color: #fff;
      background: linear-gradient(160deg, #2a7a4f 0%, #1f5f3d 100%);
      cursor: pointer;
    }
    .cta[disabled] {
      opacity: .65;
      cursor: wait;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }
    .results-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .download-zip {
      text-decoration: none;
      border: 1px solid #b4c3b8;
      background: var(--accent-soft);
      color: #21462f;
      border-radius: 8px;
      padding: 7px 10px;
      font-size: 13px;
      font-weight: 800;
    }
    .results-grid {
      margin-top: 10px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
      gap: 12px;
    }
    .result-card {
      border: 1px solid #d2dbd4;
      border-radius: 12px;
      background: #fff;
      padding: 10px;
      display: grid;
      gap: 10px;
    }
    .result-card.error {
      border-color: #ddb7b7;
      background: #fff8f8;
    }
    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
      flex-wrap: wrap;
    }
    .result-title {
      font-weight: 800;
      font-size: 14px;
      color: #13231a;
      overflow-wrap: anywhere;
    }
    .result-meta {
      font-size: 12px;
      color: var(--muted);
    }
    .download-one {
      text-decoration: none;
      font-size: 12px;
      font-weight: 800;
      color: #234f34;
      border: 1px solid #b7c5ba;
      border-radius: 7px;
      padding: 5px 8px;
      background: #f6faf7;
    }
    .viewer {
      border: 1px solid #d5ded7;
      border-radius: 10px;
      background: #f5f8f5;
      min-height: 220px;
      height: 220px;
      position: relative;
      overflow: hidden;
    }
    .viewer canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .viewer-note {
      position: absolute;
      left: 8px;
      top: 8px;
      font-size: 11px;
      color: #516257;
      background: rgba(255, 255, 255, 0.86);
      border-radius: 6px;
      padding: 2px 6px;
      border: 1px solid #d7e0d9;
      pointer-events: none;
    }
    .svg-tools {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .svg-tools label {
      font-size: 12px;
      color: #2f3f35;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 700;
    }
    .svg-tools input[type="range"] {
      width: 180px;
    }
    .svg-open {
      font-size: 12px;
      font-weight: 700;
      color: #224f33;
      text-decoration: none;
    }
    .svg-wrap {
      border: 1px solid #d3ddd5;
      border-radius: 10px;
      background: #f8faf8;
      height: 260px;
      overflow: auto;
      position: relative;
    }
    .svg-inner {
      transform-origin: 0 0;
      padding: 8px;
      width: max-content;
    }
    .svg-inner img {
      display: block;
      max-width: none;
      height: auto;
      border: 1px solid #e0e6e1;
      background: #fff;
    }
    .failure {
      color: #7a1f1f;
      font-size: 13px;
      font-weight: 600;
      line-height: 1.35;
      white-space: pre-wrap;
    }
    .foot {
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .results-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>Lego Sorter Bin Generator</h1>
      <p class="sub">Mehrere STEP-Varianten auswählen, parallel generieren, 3D/SVG prüfen und einzeln oder als ZIP herunterladen.</p>
      __ERROR_BLOCK__
      <form id="generate-form">
        <div class="grid">
          <div class="full">
            <label class="label">STEP Files</label>
            <div class="step-box">
              <div class="step-tools">
                <button type="button" class="tiny-btn" id="select-all">Alle auswählen</button>
                <button type="button" class="tiny-btn" id="select-none">Alle abwählen</button>
              </div>
              <div class="step-grid" id="step-grid">__STEP_OPTIONS__</div>
            </div>
          </div>

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
            <input id="thickness" name="thickness" type="number" min="0.1" step="0.01" value="3.2">
          </div>

          <div>
            <label class="label" for="kerf">Kerf (mm)</label>
            <input id="kerf" name="kerf" type="number" step="0.01" value="0.02">
          </div>

          <div id="packed-block" class="packed full">
            <div class="grid">
              <div>
                <label class="label" for="sheet_width">Sheet Width (mm)</label>
                <input id="sheet_width" name="sheet_width" type="number" min="1" step="1" value="710">
              </div>
              <div>
                <label class="label" for="sheet_height">Sheet Height (mm)</label>
                <input id="sheet_height" name="sheet_height" type="number" min="1" step="1" value="180">
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
        </div>

        <div class="cta-row">
          <button id="generate-btn" class="cta" type="submit">Parallel Generieren</button>
          <span id="status" class="status">Bereit.</span>
        </div>
      </form>
      <p class="foot">Hinweis: Jobs werden serverseitig temporär gespeichert und nach einigen Stunden automatisch entfernt.</p>
    </section>

    <section id="results-panel" class="panel hidden" style="margin-top: 14px;">
      <div class="results-head">
        <h2 style="margin:0;font-size:22px;">Ergebnisse</h2>
        <div id="zip-slot"></div>
      </div>
      <p id="results-meta" class="sub" style="margin-bottom: 0;"></p>
      <div id="results-grid" class="results-grid"></div>
    </section>
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
      meta.textContent = `fehlgeschlagen in ${Number(item.elapsed_s || 0).toFixed(2)}s`;
      head.appendChild(title);
      head.appendChild(meta);

      const err = document.createElement('div');
      err.className = 'failure';
      err.textContent = item.error || 'Unknown error';

      card.appendChild(head);
      card.appendChild(err);
      return card;
    }

    function initSvgZoom(slider, valueEl, inner) {
      const apply = () => {
        const zoom = Number(slider.value || 1);
        inner.style.transform = `scale(${zoom})`;
        valueEl.textContent = `${Math.round(zoom * 100)}%`;
      };
      slider.addEventListener('input', apply);
      apply();
    }

    function initMeshViewer(canvas, meshData) {
      if (!window.THREE || !meshData || !meshData.vertices || !meshData.triangles || meshData.triangles.length === 0) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const w = canvas.clientWidth || 400;
          const h = canvas.clientHeight || 220;
          canvas.width = w;
          canvas.height = h;
          ctx.fillStyle = '#f5f8f5';
          ctx.fillRect(0, 0, w, h);
          ctx.fillStyle = '#607063';
          ctx.font = '13px Segoe UI';
          ctx.fillText('3D preview unavailable', 14, 24);
        }
        return;
      }

      const THREE = window.THREE;
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf5f8f5);

      const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 20000);

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
      const solid = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          color: 0xd7e6da,
          metalness: 0.05,
          roughness: 0.86,
          side: THREE.DoubleSide,
        })
      );
      const wire = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry, 22),
        new THREE.LineBasicMaterial({ color: 0x2c6848 })
      );
      group.add(solid);
      group.add(wire);
      scene.add(group);

      const bbox = new THREE.Box3().setFromObject(solid);
      const center = bbox.getCenter(new THREE.Vector3());
      group.position.set(-center.x, -center.y, -center.z);

      const size = bbox.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z) * 0.55 || 1;
      camera.position.set(radius * 2.1, radius * 1.45, radius * 2.1);
      camera.lookAt(0, 0, 0);

      scene.add(new THREE.AmbientLight(0xffffff, 0.72));
      const key = new THREE.DirectionalLight(0xffffff, 0.58);
      key.position.set(1.8, 2.2, 1.4);
      scene.add(key);
      const rim = new THREE.DirectionalLight(0xffffff, 0.33);
      rim.position.set(-1.1, -0.8, -1.4);
      scene.add(rim);

      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      canvas.addEventListener('pointerdown', (event) => {
        dragging = true;
        lastX = event.clientX;
        lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
      });

      canvas.addEventListener('pointermove', (event) => {
        if (!dragging) return;
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        lastX = event.clientX;
        lastY = event.clientY;
        group.rotation.y += dx * 0.011;
        group.rotation.x += dy * 0.009;
      });

      canvas.addEventListener('pointerup', (event) => {
        dragging = false;
        try { canvas.releasePointerCapture(event.pointerId); } catch (_) {}
      });

      canvas.addEventListener('pointerleave', () => {
        dragging = false;
      });

      canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        const factor = event.deltaY > 0 ? 1.08 : 0.92;
        camera.position.multiplyScalar(factor);
      }, { passive: false });

      const resize = () => {
        const w = Math.max(10, canvas.clientWidth);
        const h = Math.max(10, canvas.clientHeight);
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };

      const observer = new ResizeObserver(resize);
      observer.observe(canvas);
      resize();

      function frame() {
        if (!dragging) {
          group.rotation.y += 0.0025;
        }
        renderer.render(scene, camera);
        window.requestAnimationFrame(frame);
      }
      frame();
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
      dl.textContent = 'Einzeln laden';

      right.appendChild(meta);
      right.appendChild(dl);
      head.appendChild(title);
      head.appendChild(right);

      const viewer = document.createElement('div');
      viewer.className = 'viewer';
      const note = document.createElement('div');
      note.className = 'viewer-note';
      note.textContent = '3D box preview · ziehen zum drehen · Mausrad zoom';
      const canvas = document.createElement('canvas');
      viewer.appendChild(canvas);
      viewer.appendChild(note);

      const svgTools = document.createElement('div');
      svgTools.className = 'svg-tools';
      const zoomLabel = document.createElement('label');
      zoomLabel.textContent = 'SVG Zoom';
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0.4';
      slider.max = '4';
      slider.step = '0.1';
      slider.value = '1';
      const zoomValue = document.createElement('span');
      zoomValue.textContent = '100%';
      zoomLabel.appendChild(slider);
      zoomLabel.appendChild(zoomValue);

      const open = document.createElement('a');
      open.className = 'svg-open';
      open.href = item.preview_url;
      open.target = '_blank';
      open.rel = 'noreferrer';
      open.textContent = 'SVG in neuem Tab';

      svgTools.appendChild(zoomLabel);
      svgTools.appendChild(open);

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

      card.appendChild(head);
      card.appendChild(viewer);
      card.appendChild(svgTools);
      card.appendChild(svgWrap);

      initSvgZoom(slider, zoomValue, svgInner);
      initMeshViewer(canvas, item.mesh);

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
        zip.textContent = 'ZIP herunterladen';
        zipSlot.appendChild(zip);
      }

      resultsMeta.textContent = `${payload.succeeded}/${payload.requested} erfolgreich · ${payload.failed} fehlgeschlagen · layout=${payload.layout}`;
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
        statusEl.textContent = 'Bitte mindestens eine Box auswählen.';
        return;
      }

      const formData = new FormData(form);
      statusEl.textContent = `Starte parallele Generierung für ${chosen.length} Datei(en) ...`;
      generateBtn.disabled = true;

      try {
        const resp = await fetch('/api/generate-batch', {
          method: 'POST',
          body: formData,
        });
        const payload = await parseJsonResponse(resp);
        renderResults(payload);
        statusEl.textContent = `Fertig: ${payload.succeeded}/${payload.requested} erfolgreich.`;
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        statusEl.textContent = `Fehler: ${msg}`;
      } finally {
        generateBtn.disabled = false;
      }
    });

    selectAllBtn.addEventListener('click', () => setAllStepCheckboxes(true));
    selectNoneBtn.addEventListener('click', () => setAllStepCheckboxes(false));
    layoutSel.addEventListener('change', syncLayout);
    syncLayout();
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
