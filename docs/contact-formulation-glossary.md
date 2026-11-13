# ContactFormulation terminology

Use these terms when changing ContactFormulation physics, pair evaluation and assembly, and their geometry,
quadrature, search, and numerical helpers. This glossary defines concepts and naming conventions. Method-specific
algorithms, sign conventions, and support limits belong in the documentation for those methods.

## Geometry and meshes

| Term | Meaning |
| --- | --- |
| Vertex | A geometric corner or endpoint. A quadratic edge has two vertices and three finite-element nodes. |
| Finite-element (FE) node | A location associated with a nodal interpolation coefficient. Nodes need not be vertices. |
| Contact element | An edge in two spatial dimensions or a surface face in three spatial dimensions. Existing mesh APIs may call either a face. |
| mesh1 / mesh2 | The registered `MeshData` objects, or their `MeshData::Viewer` views, in coupling-scheme order. |
| geometry1 / geometry2 | The current element geometries for one pair, in the same order as the meshes. |
| Parametric coordinates | Coordinates within an element's reference domain. Edges use xi in [-1, 1]; quadrilaterals use [-1, 1]^2; linear triangles use the simplex with vertices (0, 0), (1, 0), (0, 1). |
| Spatial coordinates | Coordinates in physical space, such as x, y, and z. An edge has parametric dimension one even when embedded in spatial dimension two. |
| Parent mesh | The host application's MFEM volume mesh. This does not mean an element's parametric coordinate system. |
| Submesh | The boundary mesh extracted from the parent mesh for contact fields and transfers. |
| LOR mesh | A low-order refined mesh used in the MFEM transfer hierarchy. |
| Redecomp mesh | The mesh redistributed for local contact search and integration. Its local numbering differs from the parent/submesh numbering. |

## Basis functions, values, and degrees of freedom

A **basis** is a collection of basis functions. Its **size** is the number of functions in that collection. A
**degree of freedom (DOF)** identifies one scalar coefficient. Basis functions may be scalar-valued or vector-valued;
their coefficients remain scalar. A vector-valued field can have several scalar DOFs at a node.

For a two-function scalar basis used for 2D displacement, each basis-function index has two values, giving four scalar
coefficients. `SizeGeom1Basis = 2` and `NumValuesPerBasis = 2` describe that element's four residual entries. A
quadratic
Lagrange edge has basis size three and two geometric vertices.

| Term | Meaning | MFEM connection for scalar nodal bases |
| --- | --- | --- |
| Basis size | The number of basis functions. | `GetElementDofs(element, dofs)` fills an array with one index per element basis function; `dofs.Size()` is that element's basis size. `GetNDofs()` counts the underlying basis coefficients across the rank's local FE space, before accounting for multiple values per basis. |
| Basis index | The element-local index selecting a basis function. Use `basis_index`. | `dofs[basis_index]`, from `GetElementDofs(element, dofs)`, gives the corresponding index in the underlying local FE space. |
| Values per basis | The number of scalar entries associated with each basis-function index. For example, 2D displacement has two values per basis function. Use `NumValuesPerBasis` / `num_values_per_basis`. | `GetVDim()` gives this number. It is 1 for a scalar field and 2 for 2D displacement. |
| Values per node | The number of scalar field entries stored at each FE node, such as x and y displacement. This describes nodal storage; values per basis describes entries associated with a basis-function index. They coincide for nodal Lagrange fields, but values per node may not apply to non-nodal bases. Use `NumValuesPerNode` / `num_values_per_node`. | For nodal Lagrange fields, `GetVDim()` also gives the number of values per node. |
| Local DOF | A coefficient index in a local FE-space vector. Shared nodes can have local entries on several MPI ranks. | `DofToVDof(dofs[basis_index], value_index)` selects one scalar coefficient in the field vector. MFEM calls this a "vector DOF" index. `GetElementVDofs()` lists these indices for an element; `GetVSize()` counts all entries in the local field vector. |
| True DOF | An independent, conforming coefficient index after shared-node and constraint relations are applied. Assembly transfers local contributions to true DOFs. | `GetTrueVSize()` gives the number of entries in the rank's true-DOF field vector. |
| Field | A specified scalar or vector quantity over an FE space. Weighted gap and tributary area are two scalar fields; x and y displacement entries belong to one vector field. | A `GridFunction` stores local field coefficients. Its size is `GetVSize() = GetNDofs() * GetVDim()`: underlying basis coefficients times values per basis. |

For the nodal Lagrange fields used here, basis functions correspond to nodes, so values per basis and values per node
coincide. They are not general synonyms. Distinguish basis-function indices from scalar DOF indices. For example,
each basis-function index in a 2D displacement field has two associated scalar DOFs.

## Names, sizes, and storage

| Term | Naming and meaning |
| --- | --- |
| Numerical extent | An array length or matrix row/column size. Use `Size...` / `size_...`, such as `SizeValues`, `size_coords`, and `size_geom1_basis`. For example, `size_coords` is the number of scalar entries in a coordinate input array. |
| Entity count | `Num...` / `num_...`, such as `num_pairs`, `num_nodes`, `num_fields`, and `num_required_points`. |
| Multiplicity | `Num...` / `num_...` for values per basis or node. |
| Dimension | `Dim` / `dim` in identifiers, including `spatial_dim` and `parametric_dim`. Spell out "dimension" in prose. |
| Rank | The number of array dimensions, returned by `rank()`. |
| Extent | The size along one array dimension, returned by `extent<Dim>()`. |
| Size | The number of entries, returned by `size()`. Fixed numerical sizes may be compile-time constants. |
| Capacity | The number of allocated entries available, which can exceed the number of populated entries. Keep `capacity()` and capacity limits distinct from completed sizes/counts. |
| Owning array | An object containing or managing its elements. `MdArray` stores fixed-size numerical data without dynamic allocation. |
| Referencing view | An object referring to existing elements. The elements must remain alive, accessible to the execution backend, and unchanged in location while the view is used. |

An array length is also a count. The `size_...` and `num_...` prefixes are a naming convention based on how a quantity
is used: `size_...` describes array lengths and matrix extents; `num_...` describes quantities such as nodes, pairs,
and values per node. They do not define mathematically distinct kinds of numbers.

Use `geometry1` / `geometry2` for objects and `Geometry1` / `Geometry2` for types; `geom1` / `geom2` are the
corresponding
abbreviations in size and coordinate-array names. `coords`, `dim`, `FE`, `DOF`, and `LM` mean coordinates, dimension,
finite element, degree of freedom, and Lagrange multiplier. External library identifiers retain their spelling.

Production pair kernels take separate coordinate arrays `coords1` and `coords2`, each indexed `[node][spatial_dim]`.
Their combined
Jacobian-column order is geometry1 coordinates followed by geometry2 coordinates. A basis-sized residual uses
`[basis_index][value_index]` within each geometry, with geometry1 entries first. Generic nodal assembly uses
`[node][value_index]`. MFEM element matrix packing converts this order to the selected FE-space ordering and
column-major matrix storage. Row and column spaces, multiplicities, and element maps are independent.

## Search and quadrature

| Term | Meaning |
| --- | --- |
| Coarse candidate | An element pair returned by spatial search. Spatial proximity alone does not establish contact. |
| Refined candidate | A coarse candidate that passes topology and geometry checks, such as opposing normals. It can still have empty quadrature or no compressive contact. |
| Quadrature overlap | The integration interval or surface determined from the current pair geometry. |
| Active pair | A successfully evaluated pair with points whose values or derivatives must be retained by the pair function. This includes integration before contact activation and does not require a nonzero current energy or residual. |
| Rule generator | A configured operation producing quadrature points and weights for current geometry. A base rule generator supplies the reference integration rule. |
| Generated rule | The resulting points, weights, and generation status. A rule is data, not its generator. |
| Integration-surface coordinates | Coordinates on the surface used to generate a rule. The mapping defines their domain and conversion to physical weights. |
| Element-parametric coordinates | Generated point locations in each element's reference domain, stored in `geom1_parametric_coords` and `geom2_parametric_coords`. |
| Physical integration weights | Quadrature weights including the geometric scaling needed to integrate over the chosen physical domain, such as its length or area. For a line segment of length L mapped from [-1, 1], each reference weight is multiplied by L/2. |

`QuadratureStatus::Success` means every reported point is initialized. Success with zero points is a valid empty rule,
not an error or proof of compression. `InsufficientCapacity` reports the complete required point count; partial data
must be discarded. `InvalidGeometry` reports zero points and unusable numerical output. Check generation status before
consuming pair values or interpreting activation flags. Acceptance by refined search and successful quadrature are
separate conditions.

## Gaps and enforcement

| Term | Meaning |
| --- | --- |
| Geometric gap | A signed measure of separation between contacting geometries. The formulation defines how it is measured and which sign denotes opening or penetration. |
| Gap offset | A prescribed shift of the geometric gap used to define the contact constraint; called `residual_gap` in configuration. |
| Weighted gap | The basis-weighted gap integral `G_i = integral(N_i * gap dS)` over the chosen integration domain, where N_i is a basis function and dS is the physical integration measure. |
| Tributary area | The basis-weighted integration measure `A_i = integral(N_i dS)`. For line contact, this quantity has units of length. |
| Normalized gap | The weighted gap divided by its tributary area, `G_i / A_i`, where A_i is nonzero. |
| Penalty stiffness | A coefficient controlling the strength of a penalty relation between contact-constraint violation and the resulting contact response. |
| Contact pressure | A scalar describing normal contact traction per unit integration measure. Its sign convention is defined by the formulation. |
| Lagrange multiplier (LM) | An additional unknown used to enforce a constraint. In normal contact, a multiplier can represent contact pressure. |
| Enforcement location | Where the contact condition is imposed, such as at quadrature points or on assembled nodal quantities. |
| Contact activation | The condition determining whether a contact constraint contributes to the calculation. |
| Contact active set | The contact constraints treated as active by the enforcement method. Pair integration can be active before this set is determined. |

## Values, derivatives, assembly, and results

For a pair function f_i(x), **values** are its scalar outputs, the **Jacobian** is `J_ij = df_i/dx_j`, and each output's
**Hessian** is `H_ijk = d2f_i/(dx_j dx_k)`. Here x contains current spatial coordinates in geometry1-then-geometry2
order.
The outputs may be energy, residual entries, or gap/area integrals. Settings are held fixed during differentiation.

**Output-weighted derivatives** contract these arrays with fixed output weights w_i: `sum_i w_i J_ij` and
`sum_i w_i H_ijk`. These weights differ from the quadrature weights already included in f. All-zero output weights
produce zero contracted derivatives. Energy partials with respect to weighted gap and tributary area are distinct from
energy derivatives with respect to spatial coordinates.

A **pair contribution** contains one pair's numerical outputs before global assembly. A **pair kernel** evaluates a
batch
into execution-accessible buffers. **Assembly** sums vector entries at shared DOFs, packs matrix blocks, transfers
between mesh levels, and accumulates results on true DOFs. An **assembled field** contains scalar field values and their
coordinate `jacobian`. A **contact residual**, exposed as `force`, is the vector of contact contributions to the
discrete equations. Its sign convention is defined by the formulation.

| Result | Meaning |
| --- | --- |
| `df_dx` | Coordinate derivative of the contact residual. For an energy formulation it is the coordinate Hessian of the energy. |
| `df_dp` | Derivative of the contact residual with respect to LM pressure values. |
| `weighted_gap_jacobian` | Coordinate Jacobian of the weighted-gap constraint, including any contact activation. It is distinct from a normalized-gap Jacobian. |
| `has_active_points` | Whether a successfully evaluated pair has active points whose values or derivatives must be retained. For nodal integrals, nonempty integration is active even for opening gaps, before contact activation. |

A function value can be zero while its derivatives are nonzero. Activation flags therefore account for derivative
outputs as well as function values, including tangents at zero gap. Kernels returning quadrature status use these flags
to zero inactive outputs before storing numerical values and derivatives; the buffers retain quadrature status
separately.

An empty optional or null output pointer means that quantity was not calculated or has been taken from the result.
A calculated zero is an available result. Returned outputs remain valid after another calculation or destruction of the
formulation, provided their associated MFEM spaces and MPI communicator remain alive. Formulations retain fixed
configuration and references to transfer data, but do not retain mesh views or previously returned outputs.
