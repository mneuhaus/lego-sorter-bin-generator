"""STEP file loading and planar face extraction."""

from dataclasses import dataclass, field

import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GCPnts import GCPnts_UniformAbscissa
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Line
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Vec, gp_Pnt


@dataclass
class EdgeData:
    """A 3D edge defined by start and end points."""
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    midpoint: tuple[float, float, float]


@dataclass
class PlanarFace:
    """A planar face extracted from a STEP model."""
    face_id: int
    normal: tuple[float, float, float]
    center: tuple[float, float, float]
    area: float
    edges: list[EdgeData] = field(default_factory=list)
    outer_wire_edges: list[EdgeData] = field(default_factory=list)
    inner_wires_edges: list[list[EdgeData]] = field(default_factory=list)
    occ_face: object = None  # raw OCC face for later use


def _point_to_tuple(pt: gp_Pnt) -> tuple[float, float, float]:
    return (pt.X(), pt.Y(), pt.Z())


def _vec_to_tuple(v: gp_Vec) -> tuple[float, float, float]:
    return (v.X(), v.Y(), v.Z())


def _extract_edges_from_wire(wire, curve_deflection: float = 1.0) -> list[EdgeData]:
    """Extract edges from an OCC wire.

    Straight edges produce a single EdgeData. Curved edges are tessellated
    into multiple short segments so curves are preserved in the 2D projection.

    Args:
        wire: OCC wire topology.
        curve_deflection: Maximum segment length in mm for tessellating curves.
    """
    edges = []
    explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        try:
            adaptor = BRepAdaptor_Curve(edge)
            u_start = adaptor.FirstParameter()
            u_end = adaptor.LastParameter()
            curve = BRep_Tool.Curve_s(edge, 0.0, 0.0)
            if curve is None:
                explorer.Next()
                continue

            if adaptor.GetType() == GeomAbs_Line:
                # Straight edge: single segment
                p1 = curve.Value(u_start)
                p2 = curve.Value(u_end)
                mid = curve.Value((u_start + u_end) / 2)
                edges.append(EdgeData(
                    start=_point_to_tuple(p1),
                    end=_point_to_tuple(p2),
                    midpoint=_point_to_tuple(mid),
                ))
            else:
                # Curved edge: tessellate into segments
                sampler = GCPnts_UniformAbscissa(adaptor, curve_deflection)
                if sampler.IsDone() and sampler.NbPoints() >= 2:
                    n_pts = sampler.NbPoints()
                    params = [sampler.Parameter(i + 1) for i in range(n_pts)]
                else:
                    # Fallback: sample at regular parameter intervals
                    import math
                    p_start = curve.Value(u_start)
                    p_end = curve.Value(u_end)
                    chord = math.sqrt(sum((a - b)**2 for a, b in
                                         zip(_point_to_tuple(p_start),
                                             _point_to_tuple(p_end))))
                    n_segs = max(8, int(chord / curve_deflection))
                    params = [u_start + i * (u_end - u_start) / n_segs
                              for i in range(n_segs + 1)]

                for i in range(len(params) - 1):
                    p1 = curve.Value(params[i])
                    p2 = curve.Value(params[i + 1])
                    mid = curve.Value((params[i] + params[i + 1]) / 2)
                    edges.append(EdgeData(
                        start=_point_to_tuple(p1),
                        end=_point_to_tuple(p2),
                        midpoint=_point_to_tuple(mid),
                    ))
        except Exception:
            pass
        explorer.Next()
    return edges


def load_step(filepath: str, min_area: float = 100.0, body_index: int = 0) -> list[PlanarFace]:
    """Load a STEP file and return planar faces above the area threshold.

    Args:
        filepath: Path to the STEP file.
        min_area: Minimum face area in mm² (default 100 = ~1 cm²).
        body_index: Which body to process (0-based). Use -1 for all bodies.

    Returns:
        List of PlanarFace objects.
    """
    assembly = cq.importers.importStep(filepath)

    all_solids = assembly.solids().vals()
    print(f"  STEP contains {len(all_solids)} solid(s)")

    if body_index >= 0:
        if body_index >= len(all_solids):
            raise ValueError(f"Body index {body_index} out of range (have {len(all_solids)} bodies)")
        solids_to_process = [all_solids[body_index]]
        print(f"  Processing body {body_index}")
    else:
        solids_to_process = all_solids
        print(f"  Processing all {len(all_solids)} bodies")

    faces = []
    face_id = 0

    for solid in solids_to_process:
        explorer = TopExp_Explorer(solid.wrapped, TopAbs_FACE)
        while explorer.More():
            occ_face = TopoDS.Face_s(explorer.Current())

            # Check if face is planar
            adaptor = BRepAdaptor_Surface(occ_face)
            if adaptor.GetType() == GeomAbs_Plane:
                plane = adaptor.Plane()

                # Compute area
                props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(occ_face, props)
                area = props.Mass()

                if area >= min_area:
                    # Get normal and center
                    normal = plane.Axis().Direction()
                    loc = plane.Location()

                    # Get center of mass
                    com = props.CentreOfMass()

                    # Extract wires (outer + inner)
                    wire_explorer = TopExp_Explorer(occ_face, TopAbs_WIRE)
                    wires = []
                    while wire_explorer.More():
                        wires.append(wire_explorer.Current())
                        wire_explorer.Next()

                    # First wire is outer, rest are inner (holes)
                    outer_edges = _extract_edges_from_wire(wires[0]) if wires else []
                    inner_wires = [_extract_edges_from_wire(w) for w in wires[1:]]
                    all_edges = outer_edges[:]
                    for iw in inner_wires:
                        all_edges.extend(iw)

                    faces.append(PlanarFace(
                        face_id=face_id,
                        normal=(normal.X(), normal.Y(), normal.Z()),
                        center=_point_to_tuple(com),
                        area=area,
                        edges=all_edges,
                        outer_wire_edges=outer_edges,
                        inner_wires_edges=inner_wires,
                        occ_face=occ_face,
                    ))
                    face_id += 1

            explorer.Next()

    return faces
