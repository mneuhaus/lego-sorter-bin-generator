"""Small web UI for generating lasercut SVG files."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.background import BackgroundTask

from lasercut.exporter import export_svg
from lasercut.joints import apply_finger_joints
from lasercut.panels import load_step_panels


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _step_dir() -> Path:
    configured = os.getenv("LASERCUT_STEP_DIR")
    if configured:
        return Path(configured)
    return _repo_root() / "step_files"


def _num_token(value: float) -> str:
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    return s or "0"


def _available_step_files() -> list[str]:
    step_dir = _step_dir()
    if not step_dir.exists():
        return []
    return sorted(
        p.name for p in step_dir.iterdir()
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


def _cleanup_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _render_index(error: str | None = None) -> str:
    options = []
    for name in _available_step_files():
        selected = ' selected="selected"' if name == "bin_half_right.step" else ""
        options.append(f'<option value="{name}"{selected}>{name}</option>')
    options_html = "\n".join(options)
    if not options_html:
        options_html = '<option value="">No STEP files found</option>'

    error_block = ""
    if error:
        error_block = f'<p class="error">{error}</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lego Sorter Bin Generator</title>
  <style>
    :root {{
      --bg: #f4f5f2;
      --panel: #ffffff;
      --text: #1d1d1d;
      --muted: #5f6661;
      --border: #d7dbd5;
      --accent: #2d6c4a;
      --error: #8a1d1d;
    }}
    body {{
      margin: 0;
      background: linear-gradient(160deg, #f2f3ef 0%, #ebefe8 100%);
      color: var(--text);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}
    .wrap {{
      max-width: 920px;
      margin: 28px auto;
      padding: 0 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 10px 28px rgba(30, 34, 30, 0.08);
    }}
    h1 {{
      margin: 0 0 4px 0;
      font-size: 26px;
      letter-spacing: 0.2px;
    }}
    .sub {{
      margin: 0 0 16px 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    label {{
      display: block;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 6px;
      color: #2b312d;
    }}
    input, select {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #c8cec7;
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 14px;
      background: #fff;
    }}
    .packed {{
      border: 1px dashed #cfd5cf;
      border-radius: 10px;
      padding: 12px;
      margin-top: 4px;
      display: none;
    }}
    .packed.visible {{
      display: block;
    }}
    button {{
      margin-top: 14px;
      border: none;
      border-radius: 8px;
      padding: 10px 16px;
      font-size: 14px;
      font-weight: 700;
      color: #fff;
      background: var(--accent);
      cursor: pointer;
    }}
    button:hover {{
      filter: brightness(1.05);
    }}
    .error {{
      margin: 0 0 12px 0;
      color: var(--error);
      font-weight: 700;
    }}
    .foot {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 760px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Lego Sorter Bin Generator</h1>
      <p class="sub">Generate jointed SVG cut plans directly from STEP files.</p>
      {error_block}
      <form method="post" action="/generate">
        <div class="grid">
          <div class="full">
            <label for="step_file">STEP File</label>
            <select id="step_file" name="step_file">{options_html}</select>
          </div>

          <div>
            <label for="layout">Layout</label>
            <select id="layout" name="layout">
              <option value="unfolded">unfolded</option>
              <option value="packed">packed</option>
            </select>
          </div>

          <div>
            <label for="finger_width">Finger Width (mm)</label>
            <input id="finger_width" name="finger_width" type="number" min="1" step="0.1" value="20.0">
          </div>

          <div>
            <label for="thickness">Thickness (mm)</label>
            <input id="thickness" name="thickness" type="number" min="0.1" step="0.01" value="3.2">
          </div>

          <div>
            <label for="kerf">Kerf (mm)</label>
            <input id="kerf" name="kerf" type="number" step="0.01" value="0.02">
          </div>

          <div id="packed-block" class="packed full">
            <div class="grid">
              <div>
                <label for="sheet_width">Sheet Width (mm)</label>
                <input id="sheet_width" name="sheet_width" type="number" min="1" step="1" value="710">
              </div>
              <div>
                <label for="sheet_height">Sheet Height (mm)</label>
                <input id="sheet_height" name="sheet_height" type="number" min="1" step="1" value="180">
              </div>
              <div>
                <label for="part_gap">Part Gap (mm)</label>
                <input id="part_gap" name="part_gap" type="number" min="0" step="0.1" value="4">
              </div>
              <div>
                <label for="sheet_gap">Sheet Gap (mm)</label>
                <input id="sheet_gap" name="sheet_gap" type="number" min="0" step="0.1" value="20">
              </div>
              <div>
                <label for="pack_rotations">Pack Rotations</label>
                <input id="pack_rotations" name="pack_rotations" type="number" min="1" step="1" value="2">
              </div>
            </div>
          </div>
        </div>

        <button type="submit">Generate SVG</button>
      </form>
      <p class="foot">The response will download an SVG file named with thickness/kerf/layout and sheet size.</p>
    </div>
  </div>
  <script>
    const layout = document.getElementById('layout');
    const packed = document.getElementById('packed-block');
    const sync = () => packed.classList.toggle('visible', layout.value === 'packed');
    layout.addEventListener('change', sync);
    sync();
  </script>
</body>
</html>"""


app = FastAPI(title="Lego Sorter Bin Generator", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_render_index())


@app.get("/healthz", response_class=JSONResponse)
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.post("/generate")
def generate(
    step_file: str = Form(...),
    thickness: float = Form(3.2),
    finger_width: float = Form(20.0),
    kerf: float = Form(0.0),
    layout: str = Form("unfolded"),
    sheet_width: str | None = Form(None),
    sheet_height: str | None = Form(None),
    part_gap: float = Form(4.0),
    sheet_gap: float = Form(20.0),
    pack_rotations: int = Form(2),
) -> FileResponse:
    if layout not in {"unfolded", "packed"}:
        raise HTTPException(status_code=400, detail="layout must be 'unfolded' or 'packed'")

    available = _available_step_files()
    if step_file not in available:
        raise HTTPException(status_code=404, detail=f"Unknown step file: {step_file}")

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

    step_path = _step_dir() / step_file
    if not step_path.exists():
        raise HTTPException(status_code=404, detail=f"STEP file not found: {step_path}")

    try:
        original_model = load_step_panels(str(step_path), thickness)
        model = apply_finger_joints(original_model, finger_width=finger_width, kerf=kerf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate joints: {exc}") from exc

    folder_name, filename = _folder_and_filename(
        step_stem=step_path.stem,
        layout=layout,
        thickness=thickness,
        kerf=kerf,
        sheet_width=parsed_sheet_width,
        sheet_height=parsed_sheet_height,
    )

    tmp_root = Path(tempfile.mkdtemp(prefix="lasercut-web-"))
    out_dir = tmp_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename

    try:
        export_svg(
            model,
            str(output_path),
            reference_model=original_model,
            layout=layout,
            sheet_width=parsed_sheet_width,
            sheet_height=parsed_sheet_height,
            part_gap=part_gap,
            sheet_gap=sheet_gap,
            pack_rotations=pack_rotations,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to export SVG: {exc}") from exc

    return FileResponse(
        output_path,
        media_type="image/svg+xml",
        filename=filename,
        background=BackgroundTask(_cleanup_dir, tmp_root),
    )
