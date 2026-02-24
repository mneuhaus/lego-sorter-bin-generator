"""Verify the rewritten parametric.py matches STEP geometry."""

import sys
sys.path.insert(0, 'src')

from lasercut.parametric import build_bin_panels, export_combined_step
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
import cadquery as cq


def get_step_bbox(step_path):
    """Get bounding box of original STEP file."""
    shape = cq.importers.importStep(str(step_path))
    occ_shape = shape.val().wrapped
    bbox = Bnd_Box()
    BRepBndLib.Add_s(occ_shape, bbox)
    return bbox.Get()


def get_model_bbox(model):
    """Get bounding box of all panels combined."""
    all_bboxes = []
    for name, panel in model.panels.items():
        bb = panel.solid.val().BoundingBox()
        all_bboxes.append((name, bb))

    if not all_bboxes:
        return None

    xmin = min(bb.xmin for _, bb in all_bboxes)
    ymin = min(bb.ymin for _, bb in all_bboxes)
    zmin = min(bb.zmin for _, bb in all_bboxes)
    xmax = max(bb.xmax for _, bb in all_bboxes)
    ymax = max(bb.ymax for _, bb in all_bboxes)
    zmax = max(bb.zmax for _, bb in all_bboxes)

    return xmin, ymin, zmin, xmax, ymax, zmax


def verify_variant(variant_name, step_path, thickness=3.2):
    """Verify a single variant."""
    print(f"\n{'='*70}")
    print(f"Verifying: {variant_name}")
    print(f"{'='*70}")

    # Build parametric model
    try:
        model = build_bin_panels(variant_name, thickness=thickness)
    except Exception as e:
        print(f"  ERROR building model: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  Panels: {', '.join(model.panels.keys())}")
    print(f"  Shared edges: {len(model.shared_edges)}")
    print(f"  Thickness: {model.thickness}mm")

    # Print panel details
    for name, panel in model.panels.items():
        bb = panel.solid.val().BoundingBox()
        print(f"\n  Panel '{name}':")
        print(f"    Normal: ({panel.normal[0]:.4f}, {panel.normal[1]:.4f}, {panel.normal[2]:.4f})")
        print(f"    Width: {panel.width:.2f}mm")
        print(f"    Height: {panel.height:.2f}mm")
        print(f"    BBox: X[{bb.xmin:.2f},{bb.xmax:.2f}] Y[{bb.ymin:.2f},{bb.ymax:.2f}] Z[{bb.zmin:.2f},{bb.zmax:.2f}]")

    # Get bounding boxes
    step_bb = get_step_bbox(step_path)
    model_bb = get_model_bbox(model)

    if model_bb is None:
        print("  ERROR: No panels generated")
        return False

    step_dx = step_bb[3] - step_bb[0]
    step_dy = step_bb[4] - step_bb[1]
    step_dz = step_bb[5] - step_bb[2]

    model_dx = model_bb[3] - model_bb[0]
    model_dy = model_bb[4] - model_bb[1]
    model_dz = model_bb[5] - model_bb[2]

    print(f"\n  STEP  bbox dims: {step_dx:.2f} x {step_dy:.2f} x {step_dz:.2f}")
    print(f"  Model bbox dims: {model_dx:.2f} x {model_dy:.2f} x {model_dz:.2f}")
    print(f"  Error (dims):    {abs(model_dx-step_dx):.2f} x {abs(model_dy-step_dy):.2f} x {abs(model_dz-step_dz):.2f}")

    print(f"\n  STEP  bbox: X[{step_bb[0]:.2f},{step_bb[3]:.2f}] Y[{step_bb[1]:.2f},{step_bb[4]:.2f}] Z[{step_bb[2]:.2f},{step_bb[5]:.2f}]")
    print(f"  Model bbox: X[{model_bb[0]:.2f},{model_bb[3]:.2f}] Y[{model_bb[1]:.2f},{model_bb[4]:.2f}] Z[{model_bb[2]:.2f},{model_bb[5]:.2f}]")

    # Check Y extent matches (should be very close)
    y_ok = abs(model_dy - step_dy) < 5.0
    print(f"\n  Y extent match: {'OK' if y_ok else 'FAIL'} (error={abs(model_dy-step_dy):.2f}mm)")

    # Export for visual inspection
    try:
        export_combined_step(model, f'output/rebuilt_{variant_name}.step')
        print(f"  Exported to output/rebuilt_{variant_name}.step")
    except Exception as e:
        print(f"  Export warning: {e}")

    return y_ok


# Run verification
variants = [
    ("bin_full", "step_files/bin_full.step"),
    ("bin_third_left", "step_files/bin_third_left.step"),
    ("bin_third_center", "step_files/bin_third_center.step"),
    ("bin_third_right", "step_files/bin_third_right.step"),
    ("bin_half_left", "step_files/bin_half_left.step"),
    ("bin_half_right", "step_files/bin_half_right.step"),
]

results = {}
for name, path in variants:
    try:
        results[name] = verify_variant(name, path)
    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results[name] = False

print(f"\n\n{'='*70}")
print("Summary")
print(f"{'='*70}")
for name, ok in results.items():
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")
