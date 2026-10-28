# FEM Optimization

This repository implements a small finite element analysis workflow for 3D linear elasticity on a structured hexahedral grid. The distinguishing idea is that the solid geometry is represented implicitly through a signed distance function rather than by meshing the volume directly into conforming finite elements.

In practical terms, the code does the following:

1. create or load a surface geometry
2. embed that geometry inside a regular Cartesian hex grid
3. classify cells as fully inside, fully outside, or cut by the boundary
4. integrate stiffness and forces on full and cut cells
5. assemble the global linear system
6. apply boundary conditions
7. solve for displacement
8. visualize the undeformed and deformed geometry

The repository is best understood as a computational pipeline rather than as one monolithic solver.

## Who This Is For

This README is written for someone who is new either to this codebase or to this style of embedded finite element workflow.

If you already know classical FEM, the main conceptual shift is this:

- the geometry boundary is described by a surface mesh and queried through a signed distance function
- the analysis grid is a separate structured hexahedral background grid
- elements intersected by the boundary are handled with special integration rather than by generating a body-fitted mesh

## Repository Layout

```text
.
├── assembly/
│   ├── __init__.py
│   ├── force.py
│   ├── global_system.py
│   ├── nitsche.py
│   └── stiffness.py
├── core/
│   ├── __init__.py
│   └── grid_indexing.py
├── integration/
│   ├── element_force.py
│   ├── element_stiffness.py
│   ├── gauss_integration.py
│   ├── nitsche_bc.py
│   └── octreesampler.py
├── notebooks/
│   ├── displacement_analysis.ipynb
│   └── stiffness_analyzis.ipynb
├── docs/
│   └── sparse_matrix_method.md
├── outputs/
│   └── README.md
├── boundary.py
├── classification.py
├── constants.py
├── fem.py
├── force_factory.py
├── geometry.py
├── kinematics.py
├── shape.py
├── tensor.py
├── visualization.py
├── volume_factory.py
└── README.md
```

## High-Level Data Flow

### 1. Geometry Definition

The analysis starts with a surface geometry. In this repository, geometry is wrapped by `Volume` in `geometry.py`.

- `Volume.mesh` stores the triangle surface mesh
- `Volume.signed_distance(points)` evaluates the signed distance at NumPy points
- `Volume.signed_distance_torch(x)` does the same for torch tensors

The sign convention in the current code is important because classification and cut-cell integration depend on it. The code treats sign queries consistently within its own pipeline, so the main rule for a newcomer is: do not change the sign convention in one place without checking classification, octree sampling, and force integration.

`volume_factory.py` provides convenience constructors:

- `VolumeFactory.sphere(...)`
- `VolumeFactory.cylinder(...)`
- `VolumeFactory.box(...)`
- `VolumeFactory.from_file(path)`

These are the easiest entry points when building example cases.

### 2. Background Grid

The analysis domain is discretized by `HexGrid` in `geometry.py`.

`HexGrid` stores:

- `bounds`: the physical bounding box
- `resolution`: number of cells in x, y, z
- `centers`: structured array of cell centers
- `dx`, `dy`, `dz`: cell sizes

This grid is not fitted to the geometry boundary. Instead, it acts as a background mesh that is later filtered by SDF classification.

### 3. Cell Classification

`classification.py` decides which background cells participate in the simulation.

`Classification.classify_sdf(volume, grid)` returns three fields:

- `inside`: cells whose eight corners are inside the geometry
- `partial`: cells cut by the boundary
- `sdf`: the signed distance evaluated at cell centers

This is the first major idea in the codebase:

- fully inside cells use standard hexahedral quadrature
- cut cells require special treatment because only part of the background cell belongs to the physical solid
- outside cells are ignored

### 4. Material Model

`constants.py` contains `Material`, which stores:

- `E`: Young's modulus
- `nu`: Poisson ratio
- `C`: the 6x6 isotropic elasticity matrix

This project currently uses small-strain linear elasticity in 3D.

### 5. Shape Functions and Kinematics

These files are the mathematical core of the hex element formulation.

`shape.py` contains:

- `Shape.N3(...)`: trilinear shape functions for an 8-node hexahedron
- `Shape.dN3_dxi(...)`: derivatives in reference coordinates
- `Shape.jacobian(...)`: geometric Jacobian
- `Shape.dN3_dx(...)`: derivatives in physical coordinates
- `Shape.N2(...)`, `Shape.dN2(...)`: bilinear surface shape functions used in surface integration helpers
- `Shape.inverse_map(...)`: inverse mapping helper from physical to reference coordinates

`kinematics.py` contains:

- `Kinematics.B_matrix(dN3_dx)`

This builds the standard strain-displacement matrix $B$, which maps nodal displacements to strains.

For a newcomer, the practical relation is:

- shape functions describe interpolation inside an element
- the Jacobian maps reference derivatives to physical derivatives
- the $B$ matrix is what finally enters the stiffness integral

### 6. Numerical Integration

There are two integration strategies in the repository.

#### Standard Gaussian quadrature

`integration/gauss_integration.py` provides:

- `Gauss.gauss_3d(order, ...)` for hexahedral volume integration
- `Gauss.gauss_2d(order, ...)` for quadrilateral surface integration support
- `Gauss.gauss_triangle(order, ...)` for triangle integration

#### Adaptive cut-cell integration

`integration/octreesampler.py` contains `OctreeSampler.sample_batch(...)`.

This module is used for cells cut by the geometry boundary. Instead of integrating over the whole hexahedron, it recursively subdivides the reference cell into smaller cubes and keeps only the parts classified as belonging to the solid.

Conceptually:

1. start with the reference cell
2. classify its subcubes using SDF values at the cube corners
3. accept fully interior cubes
4. subdivide cubes cut by the boundary
5. collect accepted sample points and weights

This is the mechanism that allows the code to approximate the contribution of irregular cut cells without a fitted body mesh.

### 7. Element Stiffness Computation

`integration/element_stiffness.py` defines `Stiffness`.

It has two main paths:

- `Stiffness.elements(...)` for regular, fully interior cells
- `Stiffness.elements_cut(...)` for cut cells using octree sampling

The mathematical core is the standard linear elasticity element integral:

$$
K_e = \int_{\Omega_e} B^T C B \, d\Omega
$$

For full cells, the integral is computed with standard Gauss points. For cut cells, it is approximated by the octree-based sampling rule.

### 8. Global Stiffness Assembly

`assembly/stiffness.py` takes the local element matrices and places them into the global matrix.

Important functions:

- `StiffnessAssembly.compute_element_matrices(...)`
- `StiffnessAssembly.assemble_global_stiffness(...)`

The first computes local element matrices only for active cells. The second maps element node connectivity into global degree-of-freedom indices and accumulates local contributions into the full system matrix.

### 9. Force Definition and Integration

There are two parts here.

`force_factory.py` defines user-facing force and traction generators:

- constant body forces
- gravity
- linear and radial fields
- constant traction
- normal pressure
- directional pressure
- masked force fields

`integration/element_force.py` defines the actual integration routines:

- `Force.integrate_surface_force(...)`
- `Force.integrate_volume_force(...)`

Then `assembly/force.py` assembles local force contributions into the global load vector through:

- `ForceAssembly.compute_element_force(...)`
- `ForceAssembly.assemble_global_force(...)`

The force pipeline mirrors the stiffness pipeline:

- local integration first
- global assembly second

### 10. Boundary Conditions

Boundary conditions in `basic_simulation` are imposed via symmetric Nitsche terms.

The assembly workflow is explicitly split into four stages:

1. Compute global force `F` via `ForceAssembly`.
2. Compute global stiffness `K` via `StiffnessAssembly`.
3. Compute `nitsche_sets` via `NitscheAssembly.compute_terms(...)`.
4. Merge all terms via `GlobalAssembly.assemble_system(...)`.

`StiffnessAssembly.assemble_global_system(...)` remains available as a compatibility wrapper, but the recommended workflow is the explicit 4-step split above.

The former strong Dirichlet helper (`Boundary.apply_dirichlet`) is disabled in this package to avoid mixing boundary-condition methods in the same workflow.

### 11. Visualization

`visualization.py` contains PyVista-based helpers.

Main capabilities:

- show the implicit geometry as an SDF iso-surface
- visualize cut cells or octree cubes
- show wireframes of sampled cells
- visualize active-node displacement clouds
- build displaced surfaces by interpolating nodal displacement to surface points

This module is diagnostic as much as it is presentational. It helps answer questions like:

- is the SDF aligned correctly with the grid?
- are the cut cells where I expect them to be?
- does the deformed geometry move in the right direction?

### 12. Small Utility Layer

`tensor.py` contains `_tensor(...)`, a small convenience wrapper that converts array-like inputs into torch tensors on the correct device and dtype.

`core/grid_indexing.py` contains structured-grid indexing helpers:

- `global_to_local(...)`
- `cell_index(...)`
- `active_element_nodes(...)`

These support element lookup and degree-of-freedom assembly logic.

### 13. Project Facade

`fem.py` exposes a facade class named `FEM` that simply groups the major components under one namespace.

This is why the notebooks can call things such as:

```python
FEM.Classification.classify_sdf(...)
FEM.StiffnessAssembly.compute_element_matrices(...)
FEM.ForceAssembly.compute_element_force(...)
```

It is a convenience layer rather than a solver class with internal state.

## What the Notebooks Do

### `notebooks/displacement_analysis.ipynb`

This is the most complete end-to-end example in the repository.

It does the following:

1. creates a cylinder geometry
2. builds a background grid
3. classifies cells with the SDF
4. creates a material
5. computes and assembles the global stiffness matrix
6. defines traction and body forces
7. computes and assembles the global force vector
8. constructs node coordinates from the structured grid
9. removes inactive nodes from the system
10. applies Dirichlet constraints on the left boundary of the grid
11. solves the reduced linear system
12. scatters the reduced displacement back to the full node set
13. visualizes the deformed surface

If you want to understand how all modules fit together in one practical workflow, start here.

### `notebooks/stiffness_analyzis.ipynb`

This notebook is more of a diagnostic and exploration notebook.

It is useful for:

- checking the stiffness pipeline on a smaller problem
- looking at eigenvalues or matrix properties
- examining cut-cell sampling behavior
- visually inspecting octree sampling and boundary regions

Despite the filename typo, it serves as a focused debugging notebook for stiffness and cut-cell integration behavior.

## End-to-End Computational Story

If you are completely new to the topic, this is the cleanest mental model:

### Step A: Represent the solid

The solid is stored as a triangle surface mesh, but the analysis mostly talks to it through a signed distance function.

Why this matters:

- geometry queries become simple point evaluations
- classification can be done by checking cell corners
- the background grid does not need to conform to the shape boundary

### Step B: Place a regular hexahedral grid around it

The grid is easy to build and index because it is structured.

Why this matters:

- connectivity is simple
- node indexing is deterministic
- assembly becomes easier than in an unstructured mesh

### Step C: Keep only the cells that belong to the solid

Cells are split into inside, outside, and partial.

Why this matters:

- outside cells are ignored
- inside cells are cheap to integrate
- partial cells preserve the geometry boundary without generating a fitted body mesh

### Step D: Compute element matrices and vectors

For every active element, the code computes:

- a local stiffness matrix
- optionally a local force vector

These are the local building blocks of the global equilibrium equation.

### Step E: Assemble the global system

Local contributions are assembled into sparse global operators and vectors.

The final algebraic problem is:

$$
K u = F
$$

where:

- $K$ is the global stiffness matrix (assembled in sparse format)
- $u$ is the nodal displacement vector
- $F$ is the global load vector

### Step F: Constrain rigid motion

Without boundary conditions, the structure can translate or rotate freely, and the linear system becomes singular.

So the code fixes certain nodes and degrees of freedom before solving.

### Step G: Solve and visualize

The displacement solution is mapped back to nodes and then visualized on the surface.

## Current Source Organization Policy

The repository is now organized with this convention:

- source code stays in the root module files and subpackages
- exploratory workflows stay in `notebooks/`
- generated non-code artifacts belong in `outputs/`

This keeps the root focused on reusable source files instead of mixed temporary artifacts.

## Recommended Reading Order For a Newcomer

If you want to understand the project in the most efficient order, read files in this sequence:

1. `README.md`
2. `geometry.py`
3. `classification.py`
4. `constants.py`
5. `shape.py`
6. `kinematics.py`
7. `integration/gauss_integration.py`
8. `integration/octreesampler.py`
9. `integration/element_stiffness.py`
10. `integration/element_force.py`
11. `assembly/stiffness.py`
12. `assembly/force.py`
13. `visualization.py`
14. `notebooks/displacement_analysis.ipynb`

That order goes from concepts and data structures to algorithms, and then to the executable workflow.

## Practical Notes and Limitations

There are a few things a newcomer should know immediately.

- The project is still research-style code rather than a polished library API.
- The main workflow now uses sparse global assembly/solve, but some exploratory notebook paths may still include dense debug experiments.
- Some routines still print debug information during integration.
- Boundary condition handling is case-specific rather than fully generic.
- The solve step currently lives in notebooks, not in a dedicated solver module.
- Visualization depends on PyVista and is mainly designed for interactive inspection.

None of these are blockers for understanding the workflow, but they explain why the repository feels more like an actively developed prototype than a packaged FEM toolkit.

## Dependencies You Should Expect

The code uses these main libraries:

- `torch` for tensor math and linear algebra
- `numpy` for array operations and quadrature setup
- `trimesh` for geometry creation, loading, and signed distance queries
- `pyvista` for visualization
- `scipy` in notebook workflows for nearest-neighbor interpolation during surface deformation

## Where To Put New Outputs

If you generate non-code results, put them in `outputs/`.

Examples:

- exported deformed meshes
- images produced from visualization runs
- CSV files of displacements or reaction forces
- notes or reports generated from notebook runs

This keeps the source tree readable and makes it obvious which files are hand-written code versus generated artifacts.
