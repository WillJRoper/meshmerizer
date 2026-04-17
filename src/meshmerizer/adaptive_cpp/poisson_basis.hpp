/**
 * @file poisson_basis.hpp
 * @brief Degree-2 quadratic B-spline basis functions and DOF infrastructure
 *        for the screened Poisson surface reconstruction (Phase 20a).
 *
 * @par References
 * - Kazhdan, M., Bolitho, M. & Hoppe, H. "Poisson Surface
 *   Reconstruction", *Proc. SGP* (2006).  Introduced the idea of
 *   solving for an indicator function whose gradient matches the
 *   smoothed normal field via a Poisson equation discretised
 *   with compactly-supported B-spline basis functions.
 * - Kazhdan, M. & Hoppe, H. "Screened Poisson Surface
 *   Reconstruction", *ACM Trans. Graph.* 32(3), Art. 29 (2013).
 *   Added a screening term that pulls the solution toward zero at
 *   sample points, improving boundary behaviour and robustness.
 * - Reference implementation: https://github.com/mkazhdan/PoissonRecon
 *   (MIT licence).  Our degree-2 B-spline choice matches the
 *   default `--degree 2` mode of this implementation.
 *
 * @par Algorithm context
 * The indicator function is expanded as
 *     chi(x) = sum_j  x_j  B_j(x)
 * where B_j are quadratic (degree-2) tensor-product B-splines
 * centred at octree leaf cell centres.  This file provides the
 * basis evaluation, gradient, DOF indexing, and stencil structure
 * needed to assemble and solve the resulting linear system.
 *
 * 1. **1-D degree-2 B-spline** evaluation and derivative.  The basis
 *    function is the quadratic B-spline with unit knot spacing,
 *    centred at 0 and with support [-3/2, 3/2] in normalised
 *    coordinates:
 *
 *        B(t) = 3/4 - t^2                   for |t| <= 1/2
 *        B(t) = (3/2 - |t|)^2 / 2           for 1/2 < |t| <= 3/2
 *        B(t) = 0                           for |t| > 3/2
 *
 * 2. **3-D tensor-product** evaluation and gradient.  Given a cell
 *    centre `c` and cell width `w`, the 3-D basis at point `x` is:
 *
 *        B_3d(x) = B((x.x - c.x) / w)
 *                * B((x.y - c.y) / w)
 *                * B((x.z - c.z) / w)
 *
 *    The gradient is obtained by replacing one factor at a time with
 *    its derivative and dividing by `w` (chain rule).
 *
 * 3. **DOF (degree-of-freedom) indexing**.  Each leaf cell in the
 *    balanced octree that lies within the B-spline support of any
 *    sample point is assigned a contiguous integer index.  The mapping
 *    is stored in two arrays:
 *
 *    - `cell_to_dof[cell_index] -> dof_index` (-1 if inactive)
 *    - `dof_to_cell[dof_index] -> cell_index`
 *
 * 4. **Stencil enumeration**.  For each DOF, we enumerate the (up to
 *    125) neighbouring DOFs whose B-spline supports overlap.  This is
 *    the sparsity pattern of the stiffness matrix.
 *
 * All functions are inline / header-only, following the project convention.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_POISSON_BASIS_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_POISSON_BASIS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "bounding_box.hpp"
#include "morton.hpp"
#include "octree_cell.hpp"
#include "vector3d.hpp"

/* ===================================================================
 * Section 1: 1-D degree-2 B-spline
 * =================================================================== */

/**
 * @brief Evaluate the 1-D degree-2 (quadratic) B-spline at normalised
 *        coordinate @p t.
 *
 * The B-spline is centred at 0 with unit knot spacing and has support
 * on [-3/2, 3/2] [SGP06, ToG13]:
 *
 *     B(t) = 3/4 - t^2                   for |t| <= 1/2
 *     B(t) = (3/2 - |t|)^2 / 2           for 1/2 < |t| <= 3/2
 *     B(t) = 0                           for |t| > 3/2
 *
 * @param t Normalised coordinate (distance from centre in units of
 *          cell width).
 * @return B-spline value in [0, 1].
 */
inline double bspline1d_evaluate(double t) {
    const double abs_t = std::fabs(t);

    /* Quadratic B-spline with unit knot spacing.
     * See Kazhdan et al. [SGP06] and Kazhdan & Hoppe [ToG13], and the
     * PoissonRecon reference implementation for the degree-2 basis.
     *
     * The compact support is |t| <= 3/2.
     * The function is C^1 (first-derivative continuous) and integrates
     * to 1 over the real line. */
    if (abs_t > 1.5) {
        return 0.0;
    }
    if (abs_t <= 0.5) {
        return 0.75 - t * t;
    }
    const double u = 1.5 - abs_t;
    return 0.5 * u * u;
}

/**
 * @brief Evaluate the derivative of the 1-D degree-2 B-spline at
 *        normalised coordinate @p t.
 *
 * The derivative (piecewise polynomial) is [SGP06, ToG13]:
 *
 *     B'(t) = -2t                        for |t| <= 1/2
 *     B'(t) = t + 3/2                    for -3/2 < t < -1/2
 *     B'(t) = t - 3/2                    for 1/2 < t < 3/2
 *     B'(t) = 0                          for |t| >= 3/2
 *
 * At the knot points (t = +/-1/2) the left and right formulas agree,
 * so we can evaluate using <= without ambiguity.  At |t| = 3/2, the
 * basis is identically zero, and we return 0.
 *
 * @param t Normalised coordinate.
 * @return Derivative value.
 */
inline double bspline1d_derivative(double t) {
    const double abs_t = std::fabs(t);

    /* Outside the compact support, the derivative vanishes. */
    if (abs_t >= 1.5) {
        return 0.0;
    }

    /* Within the central interval, the basis is 3/4 - t^2. */
    if (abs_t <= 0.5) {
        return -2.0 * t;
    }

    /* In the outer intervals, B(t) = (3/2 - |t|)^2 / 2.
     * Differentiate accounting for the absolute value. */
    return (t < 0.0) ? (t + 1.5) : (t - 1.5);
}

/* ===================================================================
 * Section 2: 3-D tensor-product B-spline
 * =================================================================== */

/**
 * @brief Evaluate the 3-D degree-2 (quadratic) B-spline centred at
 *        @p center
 *        with cell width `width`.
 *
 * The 3-D basis is the tensor product of three 1-D degree-2 B-splines
 * [SGP06, ToG13]:
 *
 *     B_3d(x) = B((x.x - c.x) / w)
 *             * B((x.y - c.y) / w)
 *             * B((x.z - c.z) / w)
 *
 * Along each axis, the 1-D support is [-3/2, 3/2] in normalised
 * coordinates, i.e. [-3/2 h, 3/2 h] in world space for cell width h.
 *
 * @param point   Evaluation point in world space.
 * @param center  Cell centre (B-spline centre).
 * @param width   Cell width (same along all axes for a cubic cell).
 * @return Basis function value in [0, 1].
 */
inline double bspline3d_evaluate(const Vector3d &point,
                                 const Vector3d &center,
                                 double width) {
    /* Normalise each axis relative to the cell centre.
     * For degree-2, the 1-D support is [-3/2 h, 3/2 h] (radius 1.5h),
     * so the normalised coordinate remains (x - c) / h. */
    const double tx = (point.x - center.x) / width;
    const double ty = (point.y - center.y) / width;
    const double tz = (point.z - center.z) / width;
    return bspline1d_evaluate(tx) *
           bspline1d_evaluate(ty) *
           bspline1d_evaluate(tz);
}

/**
 * @brief Evaluate the gradient of the 3-D degree-2 B-spline.
 *
 * Using the product rule on the tensor product [SGP06, ToG13], the
 * x-component of the gradient is:
 *
 *     dB/dx = (1/w) * B'(tx) * B(ty) * B(tz)
 *
 * and similarly for y and z.
 *
 * @param point   Evaluation point in world space.
 * @param center  Cell centre (B-spline centre).
 * @param width   Cell width.
 * @return Gradient vector {dB/dx, dB/dy, dB/dz}.
 */
inline Vector3d bspline3d_gradient(const Vector3d &point,
                                   const Vector3d &center,
                                   double width) {
    const double tx = (point.x - center.x) / width;
    const double ty = (point.y - center.y) / width;
    const double tz = (point.z - center.z) / width;

    /* Precompute the 1-D values and derivatives to avoid redundant
     * calls (each axis value is used in two gradient components). */
    const double bx = bspline1d_evaluate(tx);
    const double by = bspline1d_evaluate(ty);
    const double bz = bspline1d_evaluate(tz);

    const double dbx = bspline1d_derivative(tx);
    const double dby = bspline1d_derivative(ty);
    const double dbz = bspline1d_derivative(tz);

    /* Chain rule: d/dx B_3d = (1/w) * B'(tx) * B(ty) * B(tz). */
    const double inv_w = 1.0 / width;
    return {
        inv_w * dbx * by * bz,
        inv_w * bx * dby * bz,
        inv_w * bx * by * dbz,
    };
}

/* ===================================================================
 * Section 3: DOF indexing
 * =================================================================== */

/**
 * @brief Assign contiguous DOF indices to leaf cells in the octree.
 *
 * By default (restrict_depth < 0), every leaf cell gets a DOF.
 * When restrict_depth >= 0, only leaves at exactly that depth get
 * DOFs — this is the correct mode for adaptive octrees where the
 * Poisson stencil assumes uniform cell widths.
 *
 * The function builds two mappings:
 *
 * - `cell_to_dof`: for each cell index, the DOF index (-1 if the
 *   cell is not a leaf or is excluded by restrict_depth).
 * - `dof_to_cell`: for each DOF index, the cell index.
 *
 * DOF indices are assigned in cell-array order, which is
 * deterministic (breadth-first refinement order).
 *
 * When @p restrict_depth is non-negative, only leaves whose depth
 * equals @p restrict_depth receive DOFs.  This is critical for
 * correctness on adaptive octrees: the Poisson stencil assumes
 * uniform cell sizes (all 1-D B-spline integrals are tabulated for
 * same-scale hat functions).  Cells near the isosurface are already
 * refined to max_depth by `refine_octree`, and 2:1 balancing adds
 * neighbors at most one level coarser.  By restricting DOFs to
 * max_depth leaves only, all stencil interactions use the same cell
 * width h, making the discrete Laplacian exact.
 *
 * Coarser leaves far from the surface simply get cell_to_dof = -1
 * and are skipped by all downstream functions (find_overlapping_dofs,
 * splat_normals, accumulate_screening, apply_operator, etc.).
 *
 * @param cells          The full octree cell array.
 * @param cell_to_dof    Output mapping (resized to cells.size()).
 * @param dof_to_cell    Output mapping (resized to number of DOF
 *                       leaves).
 * @param restrict_depth If >= 0, only assign DOFs to leaves at
 *                       exactly this depth.  If < 0 (default),
 *                       all leaves get DOFs (uniform-grid mode).
 * @param depth_dof_start If non-null, filled with the starting DOF
 *                        index for each depth d (0..max_depth).
 *                        Entry d gives the first DOF index at depth d.
 *                        Has (max_depth + 2) entries: the last entry
 *                        is the total DOF count (sentinel).
 *                        Only meaningful when restrict_depth < 0.
 * @param max_depth_hint  Maximum depth in the octree.  Required when
 *                        depth_dof_start is non-null.
 */
inline void assign_dof_indices(
    const std::vector<OctreeCell> &cells,
    std::vector<std::int64_t> &cell_to_dof,
    std::vector<std::size_t> &dof_to_cell,
    int restrict_depth = -1,
    std::vector<std::int64_t> *depth_dof_start = nullptr,
    int max_depth_hint = 0) {
    cell_to_dof.assign(cells.size(), -1);
    dof_to_cell.clear();

    /* If depth-ordered assignment is requested (restrict_depth < 0
     * AND depth_dof_start is provided), we do a two-pass
     * depth-grouped assignment so DOFs at depth 0 come first, then
     * depth 1, etc.  This enables efficient depth-by-depth solving
     * in the cascadic solver (Phase 21f).
     *
     * If restrict_depth >= 0, all DOFs are at the same depth so
     * grouping is trivial. */
    const bool depth_grouped =
        (restrict_depth < 0 && depth_dof_start != nullptr);

    if (depth_grouped) {
        const int D = max_depth_hint;
        depth_dof_start->assign(
            static_cast<std::size_t>(D + 2), 0);

        /* Count leaves per depth. */
        std::vector<std::size_t> count_per_depth(
            static_cast<std::size_t>(D + 1), 0);
        for (const auto &c : cells) {
            if (c.is_leaf && c.depth <= static_cast<uint32_t>(D)) {
                ++count_per_depth[c.depth];
            }
        }

        /* Build prefix sums for starting DOF index per depth. */
        std::int64_t offset = 0;
        for (int d = 0; d <= D; ++d) {
            (*depth_dof_start)[static_cast<std::size_t>(d)] =
                offset;
            offset += static_cast<std::int64_t>(
                count_per_depth[static_cast<std::size_t>(d)]);
        }
        (*depth_dof_start)[static_cast<std::size_t>(D + 1)] =
            offset;

        /* Allocate dof_to_cell with total count. */
        const auto total = static_cast<std::size_t>(offset);
        dof_to_cell.resize(total);

        /* Assign DOFs depth-by-depth using running cursors. */
        std::vector<std::int64_t> cursor(
            depth_dof_start->begin(),
            depth_dof_start->begin() + D + 1);
        for (std::size_t i = 0; i < cells.size(); ++i) {
            if (cells[i].is_leaf &&
                cells[i].depth <= static_cast<uint32_t>(D)) {
                const auto d = cells[i].depth;
                const auto dof_idx = cursor[d];
                cell_to_dof[i] = dof_idx;
                dof_to_cell[static_cast<std::size_t>(dof_idx)] = i;
                ++cursor[d];
            }
        }
        return;
    }

    /* Original path: flat assignment (optionally restricted). */

    /* First pass: count eligible leaves so we can reserve. */
    std::size_t n_eligible = 0;
    for (const auto &c : cells) {
        if (c.is_leaf) {
            if (restrict_depth < 0 ||
                static_cast<int>(c.depth) == restrict_depth) {
                ++n_eligible;
            }
        }
    }
    dof_to_cell.reserve(n_eligible);

    /* Second pass: assign indices. */
    std::int64_t next_dof = 0;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].is_leaf) {
            if (restrict_depth < 0 ||
                static_cast<int>(cells[i].depth) ==
                    restrict_depth) {
                cell_to_dof[i] = next_dof;
                dof_to_cell.push_back(i);
                ++next_dof;
            }
        }
    }

    /* Fill depth_dof_start for single-depth case if provided. */
    if (depth_dof_start != nullptr && restrict_depth >= 0) {
        const int D = max_depth_hint > 0
                          ? max_depth_hint
                          : restrict_depth;
        depth_dof_start->assign(
            static_cast<std::size_t>(D + 2), next_dof);
        for (int d = 0; d <= D; ++d) {
            if (d < restrict_depth) {
                (*depth_dof_start)[static_cast<std::size_t>(d)] =
                    0;
            } else {
                (*depth_dof_start)[static_cast<std::size_t>(d)] =
                    (d == restrict_depth) ? 0 : next_dof;
            }
        }
        (*depth_dof_start)[static_cast<std::size_t>(D + 1)] =
            next_dof;
    }
}

/* ===================================================================
 * Section 4: Stencil enumeration (neighbor DOFs)
 * =================================================================== */

/**
 * @brief Leaf spatial index for fast neighbor lookup.
 *
 * This is a lightweight spatial hash that maps quantised cell-centre
 * positions to cell indices.  It supports hierarchical lookup for
 * cells at different depths (due to 2:1 balancing, neighbours may
 * differ by at most one depth level).
 *
 * This duplicates some logic from BalanceSpatialHash and
 * LeafSpatialIndex in other headers, but is kept self-contained here
 * for clarity and to avoid coupling the Poisson code to the dual
 * contouring code path.
 */
struct PoissonLeafHash {
    /**
     * @brief Hash function for 64-bit packed grid keys.
     */
    struct KeyHash {
        std::size_t operator()(std::uint64_t key) const {
            std::uint64_t h = key;
            h ^= h >> 33U;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33U;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33U;
            return static_cast<std::size_t>(h);
        }
    };

    std::unordered_map<std::uint64_t, std::size_t, KeyHash> map;
    std::uint32_t max_depth;
    double inv_cell_size_x;
    double inv_cell_size_y;
    double inv_cell_size_z;
    Vector3d domain_min;

    /**
     * @brief Pack three grid coordinates into a 64-bit key.
     *
     * Each axis gets 21 bits (matching the Morton encoding capacity).
     */
    static std::uint64_t pack(std::uint32_t ix, std::uint32_t iy,
                              std::uint32_t iz) {
        return (static_cast<std::uint64_t>(ix) << 42U) |
               (static_cast<std::uint64_t>(iy) << 21U) |
               static_cast<std::uint64_t>(iz);
    }

    /**
     * @brief Build the hash from all leaf cells.
     *
     * Leaf cells are keyed by their min-corner quantised to the
     * finest grid resolution (max_depth level).
     *
     * @param cells           Full octree cell array.
     * @param domain          Simulation domain bounding box.
     * @param md              Maximum octree depth.
     * @param base_resolution Top-level cells per axis.
     */
    void build(const std::vector<OctreeCell> &cells,
               const BoundingBox &domain,
               std::uint32_t md,
               std::uint32_t base_resolution) {
        max_depth = md;
        domain_min = domain.min;

        const double fine_per_axis =
            static_cast<double>(base_resolution) *
            static_cast<double>(1U << max_depth);
        inv_cell_size_x =
            fine_per_axis / (domain.max.x - domain.min.x);
        inv_cell_size_y =
            fine_per_axis / (domain.max.y - domain.min.y);
        inv_cell_size_z =
            fine_per_axis / (domain.max.z - domain.min.z);

        map.clear();
        std::size_t n_leaves = 0;
        for (const auto &c : cells) {
            if (c.is_leaf) ++n_leaves;
        }
        map.reserve(n_leaves);

        for (std::size_t i = 0; i < cells.size(); ++i) {
            if (!cells[i].is_leaf) continue;
            std::uint32_t gx, gy, gz;
            quantize(cells[i].bounds.min, gx, gy, gz);
            map[pack(gx, gy, gz)] = i;
        }
    }

    /**
     * @brief Quantize a world-space position to fine-grid coords.
     */
    void quantize(const Vector3d &pos,
                  std::uint32_t &gx,
                  std::uint32_t &gy,
                  std::uint32_t &gz) const {
        gx = static_cast<std::uint32_t>(
            (pos.x - domain_min.x) * inv_cell_size_x + 0.5);
        gy = static_cast<std::uint32_t>(
            (pos.y - domain_min.y) * inv_cell_size_y + 0.5);
        gz = static_cast<std::uint32_t>(
            (pos.z - domain_min.z) * inv_cell_size_z + 0.5);
    }

    /**
     * @brief Quantize a world-space position to fine-grid coords
     *        using the cell centre (half-cell offset from min).
     */
    void quantize_center(const Vector3d &center, double cell_width,
                         std::uint32_t &gx,
                         std::uint32_t &gy,
                         std::uint32_t &gz) const {
        /* The min corner of a cell whose centre is `center` with
         * width `cell_width` is `center - cell_width / 2`. */
        const Vector3d min_corner = {
            center.x - 0.5 * cell_width,
            center.y - 0.5 * cell_width,
            center.z - 0.5 * cell_width,
        };
        quantize(min_corner, gx, gy, gz);
    }

    /**
     * @brief Find the leaf cell containing a given fine-grid
     *        position via hierarchical probe.
     *
     * @return Cell index, or SIZE_MAX if not found.
     */
    std::size_t find_leaf_at(std::uint32_t ix, std::uint32_t iy,
                             std::uint32_t iz) const {
        for (std::uint32_t k = 0; k <= max_depth; ++k) {
            const std::uint32_t mask = ~((1U << k) - 1U);
            const std::uint64_t key =
                pack(ix & mask, iy & mask, iz & mask);
            auto it = map.find(key);
            if (it != map.end()) {
                return it->second;
            }
        }
        return SIZE_MAX;
    }
};

/**
 * @brief Enumerate the DOF stencil (neighbor DOFs) for each DOF.
 *
 * For degree-2 B-splines, two basis functions overlap if their cell
 * centres are within two cell widths in each axis direction (the
 * supports touch at three cell widths, but only at a measure-zero set).
 * On a uniform grid this gives a 5^3 = 125 stencil (including self).
 * On an adaptive grid with 2:1 balance, the stencil can be smaller at
 * depth boundaries.
 *
 * We use a spatial hash to look up neighbors efficiently.  For each
 * DOF, we probe the 125 positions offset by {-2, -1, 0, +1, +2} cell
 * widths along each axis from the cell centre.  Due to 2:1 balance, a
 * neighbor may be at the same depth or one level coarser.  Probing
 * at the fine grid level and using hierarchical lookup handles both
 * cases.
 *
 * The output is a CSR-like pair of arrays:
 *
 * - `stencil_offsets[i]` .. `stencil_offsets[i+1]` gives the range
 *   of neighbor DOF indices for DOF `i`.
 * - `stencil_neighbors[stencil_offsets[i] .. stencil_offsets[i+1]]`
 *   contains the neighbor DOF indices.
 *
 * @param cells            Full octree cell array.
 * @param cell_to_dof      Mapping from cell index to DOF index.
 * @param dof_to_cell      Mapping from DOF index to cell index.
 * @param domain           Simulation domain bounding box.
 * @param base_resolution  Top-level cells per axis.
 * @param max_depth        Maximum octree depth.
 * @param stencil_offsets  Output CSR offsets (size = n_dofs + 1).
 * @param stencil_neighbors Output neighbor DOF indices.
 */
inline void enumerate_stencils(
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const std::vector<std::size_t> &dof_to_cell,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t max_depth,
    std::vector<std::size_t> &stencil_offsets,
    std::vector<std::int64_t> &stencil_neighbors) {
    const std::size_t n_dofs = dof_to_cell.size();
    stencil_offsets.clear();
    stencil_offsets.reserve(n_dofs + 1);
    stencil_neighbors.clear();
    /* Typical stencil has ~125 entries per DOF; pre-allocate
     * conservatively. */
    stencil_neighbors.reserve(n_dofs * 125);

    /* Build a spatial hash of leaf cells for neighbor lookup. */
    PoissonLeafHash hash;
    hash.build(cells, domain, max_depth, base_resolution);

    /* Compute the finest cell width (at max_depth) for converting
     * cell-width offsets to fine-grid offsets. */
    const double domain_width_x =
        domain.max.x - domain.min.x;

    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        stencil_offsets.push_back(stencil_neighbors.size());

        const std::size_t ci = dof_to_cell[dof];
        const OctreeCell &cell = cells[ci];

        /* The cell's span in fine-grid units. */
        const std::uint32_t span =
            1U << (max_depth - cell.depth);

        /* Quantize the cell's min corner to fine-grid coords. */
        std::uint32_t gx, gy, gz;
        hash.quantize(cell.bounds.min, gx, gy, gz);

        /* Probe all 125 neighbor positions.
         * Each offset is one cell-width step in fine-grid units
         * (= span). */
        for (int dx = -2; dx <= 2; ++dx) {
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dz = -2; dz <= 2; ++dz) {
                    const std::int64_t px =
                        static_cast<std::int64_t>(gx) +
                        static_cast<std::int64_t>(dx) *
                            static_cast<std::int64_t>(span);
                    const std::int64_t py =
                        static_cast<std::int64_t>(gy) +
                        static_cast<std::int64_t>(dy) *
                            static_cast<std::int64_t>(span);
                    const std::int64_t pz =
                        static_cast<std::int64_t>(gz) +
                        static_cast<std::int64_t>(dz) *
                            static_cast<std::int64_t>(span);

                    /* Skip probes outside the domain. */
                    if (px < 0 || py < 0 || pz < 0) continue;

                    const std::size_t neighbor_ci =
                        hash.find_leaf_at(
                            static_cast<std::uint32_t>(px),
                            static_cast<std::uint32_t>(py),
                            static_cast<std::uint32_t>(pz));

                    if (neighbor_ci == SIZE_MAX) continue;

                    const std::int64_t neighbor_dof =
                        cell_to_dof[neighbor_ci];
                    if (neighbor_dof < 0) continue;

                    /* Avoid duplicate entries: only add if not
                     * already present in this DOF's stencil.
                     * Since the stencil is small (<=125), a
                     * linear scan is fine. */
                    const std::size_t start =
                        stencil_offsets.back();
                    bool found = false;
                    for (std::size_t k = start;
                         k < stencil_neighbors.size(); ++k) {
                        if (stencil_neighbors[k] ==
                            neighbor_dof) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        stencil_neighbors.push_back(
                            neighbor_dof);
                    }
                }
            }
        }
    }
    /* Final sentinel. */
    stencil_offsets.push_back(stencil_neighbors.size());
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_POISSON_BASIS_HPP_
