from dataclasses import dataclass
from typing import List

import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmsh as gmshio  # dolfin v0.10.0 API breaking change
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl

try:
    from dolfinx import default_scalar_type
except Exception:
    default_scalar_type = PETSc.ScalarType

import pyvista as pv


@dataclass
class Hole:
    x: float
    y: float
    r: float


@dataclass
class RectWithHoles:
    width: float = 2.0
    height: float = 1.0
    holes: List[Hole] = None


def sample_geometry(
    plate_width,
    plate_height,
    min_r_frac,
    max_r_frac,
    edge_margin,
    min_gap,
    max_holes,
):
    n_holes = np.random.randint(5, max_holes + 1)
    min_dim = min(plate_width, plate_height)
    r_min = min_r_frac * min_dim
    r_max = max_r_frac * min_dim

    holes = []
    attempts = 0
    max_attempts = 5000

    while len(holes) < n_holes and attempts < max_attempts:
        attempts += 1

        # sample a radius
        r = np.random.uniform(r_min, r_max)

        # available sampling box ensuring the circle stays inside the rectangle with edge_margin
        x_lo = r + edge_margin
        x_hi = plate_width - (r + edge_margin)
        y_lo = r + edge_margin
        y_hi = plate_height - (r + edge_margin)
        if x_lo >= x_hi or y_lo >= y_hi:
            continue

        # sample a candidate center
        x = np.random.uniform(x_lo, x_hi)
        y = np.random.uniform(y_lo, y_hi)

        # check overlap with existing holes
        ok = True
        for h in holes:
            dx = x - h.x
            dy = y - h.y
            if np.hypot(dx, dy) < (r + h.r + min_gap):
                ok = False
                break

        if ok:
            holes.append(Hole(x=x, y=y, r=r))

    return RectWithHoles(width=plate_width, height=plate_height, holes=holes)


def sample_mesh_params_from_fidelity(
    lc_min_coarse,
    lc_min_fine,
    band_fac_coarse,
    band_fac_fine,
    bulk_multiplicator,
    fidelity,
):
    # assert (fidelity >= 0) and (fidelity <= 1), "Fidelity must be in [0, 1]"

    lc_min = lc_min_coarse + fidelity * (lc_min_fine - lc_min_coarse)
    lc_bulk = bulk_multiplicator * lc_min

    band_fac = band_fac_coarse + fidelity * (band_fac_fine - band_fac_coarse)

    return {
        "lc_min": lc_min,
        "lc_bulk": lc_bulk,
        "band_fac": band_fac,
    }


def build_mesh(rectangle, lc_min, lc_bulk, band_fac, order, print_to_terminal=False):
    gmsh.initialize()
    if print_to_terminal:
        gmsh.option.setNumber("General.Terminal", 1)
    else:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.model.remove()  # remove model if something already exists
    gmsh.model.add("rect_with_holes")
    model, occ = gmsh.model, gmsh.model.occ

    # create rectangle and holes
    rect_tag = occ.addRectangle(0.0, 0.0, 0.0, rectangle.width, rectangle.height)
    disk_tags = [occ.addDisk(h.x, h.y, 0.0, h.r, h.r) for h in (rectangle.holes or [])]
    if disk_tags:
        occ.cut(
            objectDimTags=[(2, rect_tag)],
            toolDimTags=[(2, d) for d in disk_tags],
            removeObject=True,
            removeTool=True,
        )
    occ.synchronize()

    # single surface check
    surfs = model.getEntities(dim=2)
    assert len(surfs) == 1, f"Expected 1 surface, got {len(surfs)}"
    surf_tag = surfs[0][1]

    # set physical group and name for surface
    plate_tag = 1
    model.addPhysicalGroup(2, [surf_tag], tag=plate_tag)
    model.setPhysicalName(2, plate_tag, "Plate")

    # get boundary edge and hole curves
    tol = 1e-6 * max(rectangle.width, rectangle.height)
    sides = {"Left": (2, []), "Right": (3, []), "Bottom": (4, []), "Top": (5, [])}
    hole_curves = []
    for dim, tag in model.getBoundary([(2, surf_tag)], oriented=False):
        x, y, _ = model.occ.get_center_of_mass(dim, tag)
        if x < tol:
            sides["Left"][1].append(tag)
        elif abs(x - rectangle.width) < tol:
            sides["Right"][1].append(tag)
        elif y < tol:
            sides["Bottom"][1].append(tag)
        elif abs(y - rectangle.height) < tol:
            sides["Top"][1].append(tag)
        else:
            hole_curves.append(tag)

    # create physical groups and names for boundaries and holes
    for side_name, (side_tag, side_curves) in sides.items():
        model.addPhysicalGroup(1, side_curves, tag=side_tag)
        model.setPhysicalName(1, side_tag, side_name)
    hole_offset = 100
    for i, h_tag in enumerate(hole_curves, start=hole_offset):
        model.addPhysicalGroup(1, [h_tag], tag=i)
        model.setPhysicalName(1, i, f"Hole_{i}")

    # mesh size fields for refinement near holes and boundaries
    fields = gmsh.model.mesh.field
    f_list = []
    if hole_curves:
        distance_field = fields.add("Distance")
        fields.setNumbers(distance_field, "CurvesList", hole_curves)
        threshold_field = fields.add("Threshold")
        fields.setNumber(threshold_field, "InField", distance_field)
        fields.setNumber(threshold_field, "SizeMin", lc_min)
        fields.setNumber(threshold_field, "SizeMax", lc_bulk)
        fields.setNumber(threshold_field, "DistMin", 0.0)
        fields.setNumber(threshold_field, "DistMax", max(lc_bulk, lc_min) * band_fac)
        fields.setNumber(threshold_field, "StopAtDistMax", 1)
        f_list.append(threshold_field)
    if sides["Left"][1]:
        fd = fields.add("Distance")
        fields.setNumbers(fd, "CurvesList", sides["Left"][1])
        ft = fields.add("Threshold")
        fields.setNumber(ft, "InField", fd)
        fields.setNumber(ft, "SizeMin", lc_min * 1.5)
        fields.setNumber(ft, "SizeMax", lc_bulk)
        fields.setNumber(ft, "DistMin", 0.0)
        fields.setNumber(ft, "DistMax", 0.05 * max(rectangle.width, rectangle.height))
        fields.setNumber(ft, "StopAtDistMax", 1)
        f_list.append(ft)
    if sides["Right"][1]:
        fd = fields.add("Distance")
        fields.setNumbers(fd, "CurvesList", sides["Right"][1])
        ft = fields.add("Threshold")
        fields.setNumber(ft, "InField", fd)
        fields.setNumber(ft, "SizeMin", lc_min * 1.5)
        fields.setNumber(ft, "SizeMax", lc_bulk)
        fields.setNumber(ft, "DistMin", 0.0)
        fields.setNumber(ft, "DistMax", 0.05 * max(rectangle.width, rectangle.height))
        fields.setNumber(ft, "StopAtDistMax", 1)
        f_list.append(ft)

    # Combine using minimum
    if f_list:
        fmin = fields.add("Min")
        fields.setNumbers(fmin, "FieldsList", f_list)
        fields.setAsBackgroundMesh(fmin)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_bulk)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # delauny triangulation
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 4) # Forces at least 36 nodes on any circle

    # generate mesh and set order
    model.mesh.generate(2)
    model.mesh.setOrder(order)

    mesh = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )
    gmsh.finalize()

    return mesh


def project_to_cg1(expression, domain, tensor_shape=()):
    if tensor_shape == ():
        # scalar
        V = fem.functionspace(domain, ("Lagrange", 1))
    else:
        # tensor/vector
        V = fem.functionspace(domain, ("Lagrange", 1, tensor_shape))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # solve standard L2 projection
    problem = LinearProblem(
        ufl.inner(u, v) * ufl.dx,
        ufl.inner(expression, v) * ufl.dx,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="projection_",
    )
    return problem.solve()


def save_image(grid, var_name, min_val, max_val, img_path):
    # also save a png image of the vm stress
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.add_mesh(
        grid,
        scalars=var_name,
        cmap="viridis",
        show_edges=False,
        show_scalar_bar=False,
        clim=[min_val, max_val],
    )
    plotter.add_scalar_bar(
        title="Temperature",
        vertical=True,
        title_font_size=26,
        label_font_size=20,
        n_labels=5,
        fmt="%.1e",
        position_x=0.8,
        position_y=0.3,
        width=0.08,
        height=0.4,
    )
    plotter.view_xy()
    plotter.screenshot(img_path, scale=2)
    plotter.close()
