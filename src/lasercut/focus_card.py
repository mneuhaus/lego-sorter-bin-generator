"""Generate a black/white camera focus card as a colored 3MF."""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import cadquery as cq


CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


def _write_colored_3mf(
    output_path: str,
    solids: list[tuple[cq.Shape, str, str]],
    tolerance: float,
    angular_tolerance: float,
) -> str:
    """Write a simple colored 3MF from pre-built solids."""

    model = ET.Element(
        "model",
        {"xml:lang": "en-US", "xmlns": CORE_NS},
        unit="millimeter",
    )
    ET.SubElement(model, "metadata", name="Application").text = "lasercut-generator"
    ET.SubElement(model, "metadata", name="CreationDate").text = datetime.now().isoformat()

    resources = ET.SubElement(model, "resources")
    basematerials = ET.SubElement(resources, "basematerials", id="1")
    for _shape, color_name, color_hex in solids:
        ET.SubElement(
            basematerials,
            "base",
            name=color_name,
            displaycolor=color_hex,
        )

    for idx, (shape, _color_name, _color_hex) in enumerate(solids, start=2):
        tess = shape.tessellate(tolerance, angular_tolerance)
        verts, tris = tess
        if not verts or not tris:
            continue

        obj = ET.SubElement(
            resources,
            "object",
            id=str(idx),
            name=f"focus-card-part-{idx}",
            type="model",
            pid="1",
            pindex=str(idx - 2),
        )
        mesh = ET.SubElement(obj, "mesh")
        vertices = ET.SubElement(mesh, "vertices")
        for v in verts:
            ET.SubElement(
                vertices,
                "vertex",
                x=f"{v.x:.6f}",
                y=f"{v.y:.6f}",
                z=f"{v.z:.6f}",
            )

        triangles = ET.SubElement(mesh, "triangles")
        for t in tris:
            ET.SubElement(
                triangles,
                "triangle",
                v1=str(t[0]),
                v2=str(t[1]),
                v3=str(t[2]),
            )

    build = ET.SubElement(model, "build")
    for idx in range(2, 2 + len(solids)):
        ET.SubElement(build, "item", objectid=str(idx))

    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Relationships xmlns="{REL_NS}">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel-1" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Types xmlns="{CT_NS}">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" '
        'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        "</Types>"
    )
    model_xml = ET.tostring(model, xml_declaration=True, encoding="utf-8")

    try:
        compression = ZIP_DEFLATED
    except Exception:
        compression = ZIP_STORED

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with ZipFile(output_path, "w", compression) as zf:
        zf.writestr("_rels/.rels", rels)
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("3D/3dmodel.model", model_xml)

    return output_path


def _build_starburst_layer(size_mm: float, layer_thickness_mm: float, spokes: int) -> cq.Workplane:
    """Create alternating radial wedges clipped to the card square."""
    radius = size_mm * 1.8
    total_segments = spokes * 2  # black and white alternating
    pattern = None

    for i in range(0, total_segments, 2):
        a0 = 2.0 * math.pi * i / total_segments
        a1 = 2.0 * math.pi * (i + 1) / total_segments
        pts = [
            (0.0, 0.0),
            (radius * math.cos(a0), radius * math.sin(a0)),
            (radius * math.cos(a1), radius * math.sin(a1)),
        ]
        wedge = cq.Workplane("XY").polyline(pts).close().extrude(layer_thickness_mm)
        pattern = wedge if pattern is None else pattern.union(wedge)

    if pattern is None:
        raise ValueError("Failed to create starburst pattern")

    card_layer = cq.Workplane("XY").rect(size_mm, size_mm).extrude(layer_thickness_mm)
    return pattern.intersect(card_layer)


def generate_focus_card_3mf(
    output_path: str,
    size_mm: float = 80.0,
    thickness_mm: float = 2.0,
    top_layer_mm: float = 0.4,
    spokes: int = 16,
    tolerance: float = 0.08,
    angular_tolerance: float = 0.1,
) -> str:
    """Create a two-color focus card using a radial starburst pattern."""
    if size_mm <= 0:
        raise ValueError("size_mm must be > 0")
    if thickness_mm <= 0:
        raise ValueError("thickness_mm must be > 0")
    if top_layer_mm <= 0 or top_layer_mm >= thickness_mm:
        raise ValueError("top_layer_mm must be > 0 and < thickness_mm")
    if spokes < 4:
        raise ValueError("spokes must be >= 4")

    card = cq.Workplane("XY").rect(size_mm, size_mm).extrude(thickness_mm)
    black_top = _build_starburst_layer(size_mm, top_layer_mm, spokes).translate(
        (0.0, 0.0, thickness_mm - top_layer_mm)
    )
    white_base = card.cut(black_top)

    white_shape = white_base.val()
    black_shape = black_top.val()
    return _write_colored_3mf(
        output_path=output_path,
        solids=[
            (white_shape, "White", "#FFFFFF"),
            (black_shape, "Black", "#000000"),
        ],
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a black/white 3MF focus card.")
    parser.add_argument(
        "--output",
        default="output/focus_card_80x80x2mm_bw.3mf",
        help="Output 3MF path",
    )
    parser.add_argument("--size-mm", type=float, default=80.0, help="Card width/height in mm")
    parser.add_argument("--thickness-mm", type=float, default=2.0, help="Card thickness in mm")
    parser.add_argument(
        "--top-layer-mm",
        type=float,
        default=0.4,
        help="Thickness of the black starburst layer in mm",
    )
    parser.add_argument("--spokes", type=int, default=16, help="Number of black spokes")
    parser.add_argument("--tolerance", type=float, default=0.08, help="Mesh tolerance")
    parser.add_argument(
        "--angular-tolerance",
        type=float,
        default=0.1,
        help="Angular mesh tolerance",
    )
    args = parser.parse_args()

    out = generate_focus_card_3mf(
        output_path=args.output,
        size_mm=args.size_mm,
        thickness_mm=args.thickness_mm,
        top_layer_mm=args.top_layer_mm,
        spokes=args.spokes,
        tolerance=args.tolerance,
        angular_tolerance=args.angular_tolerance,
    )
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
