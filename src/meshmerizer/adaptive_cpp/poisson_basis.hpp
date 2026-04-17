/**
 * @file poisson_basis.hpp
 * @brief Degree-1 trilinear B-spline basis functions and DOF infrastructure
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
 *   (MIT licence).  Our degree-1 B-spline choice matches the
 *   default `--degree 1` mode of this implementation.
 *
 * @par Algorithm context
 * The indicator function is expanded as
 *     chi(x) = sum_j  x_j  B_j(x)
 * where B_j are trilinear (degree-1) tensor-product B-splines
 * centred at octree leaf cell centres.  This file provides the
 * basis evaluation, gradient, DOF indexing, and stencil structure
 * needed to assemble and solve the resulting linear system.
 *
 * 1. **1-D degree-1 B-spline** evaluation and derivative.  The basis
 *    function is the standard hat function with support [-1, 1] in
 *    normalised coordinates:
 *
 *        B(t) = max(0, 1 - |t|)
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
 *    27) neighbouring DOFs whose B-spline supports overlap.  This is
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
 * Section 1: 1-D degree-1 B-spline
 * =================================================================== */

/**
 * @brief Evaluate the 1-D degree-1 (hat) B-spline at normalised
 *        coordinate `t`.
 *
 * The B-spline is centred at 0 with support on [-1, 1]:
 *
 *     B(t) = max(0, 1 - |t|)
 *
 * @param t Normalised coordinate (distance from centre in units of
 *          cell width).
 * @return B-spline value in [0, 1].
 */
inline double bspline1d_evaluate(double t) {
    const double abs_t = std::fabs(t);
    if (abs_t >= 1.0) {
        return 0.0;
    }
    return 1.0 - abs_t;
}

/**
 * @brief Evaluate the derivative of the 1-D degree-1 B-spline at
 *        normalised coordinate `t`.
 *
 * The derivative is:
 *
 *     B'(t) = -1  if  0 < t < 1
 *              1  if -1 < t < 0
 *              0  if |t| >= 1
 *
 * At `t = 0` the hat function has a cusp; we return 0 (the average
 * of left and right derivatives) which is sufficient for Galerkin
 * integration where measure-zero points do not affect integrals.
 *
 * @param t Normalised coordinate.
 * @return Derivative value in {-1, 0, 1}.
 */
inline double bspline1d_derivative(double t) {
    const double abs_t = std::fabs(t);
    if (abs_t >= 1.0 || abs_t == 0.0) {
        return 0.0;
    }
    return (t > 0.0) ? -1.0 : 1.0;
}

/* ===================================================================
 * Section 2: 3-D tensor-product B-spline
 * =================================================================== */

/**
 * @brief Evaluate the 3-D trilinear B-spline centred at `center`
 *        with cell width `width`.
 *
 * The 3-D basis is the tensor product of three 1-D hat functions:
 *
 *     B_3d(x) = B((x.x - c.x) / w)
 *             * B((x.y - c.y) / w)
 *             * B((x.z - c.z) / w)
 *
 * @param point   Evaluation point in world space.
 * @param center  Cell centre (B-spline centre).
 * @param width   Cell width (same along all axes for a cubic cell).
 * @return Basis function value in [0, 1].
 */
inline double bspline3d_evaluate(const Vector3d &point,
                                 const Vector3d &center,
                                 double width) {
    /* Normalise each axis to the range [-1, 1] relative to the
     * cell centre.  The B-spline has support width = 2 * cell_width
     * (one cell on each side), so the normalised coordinate is
     * (x - c) / w. */
    const double tx = (point.x - center.x) / width;
    const double ty = (point.y - center.y) / width;
    const double tz = (point.z - center.z) / width;
    return bspline1d_evaluate(tx) *
           bspline1d_evaluate(ty) *
           bspline1d_evaluate(tz);
}

/**
 * @brief Evaluate the gradient of the 3-D trilinear B-spline.
 *
 * Using the product rule on the tensor product, the x-component
 * of the gradient is:
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
 * @brief Assign contiguous DOF indices to all leaf cells in the
 *        octree.
 *
 * Every leaf cell gets a DOF.  The Poisson solve needs basis
 * functions everywhere the solution could be nonzero, which
 * includes all leaves in the balanced octree (the narrow-band
 * halo is handled by the octree refinement + balance steps that
 * already ran).
 *
 * The function builds two mappings:
 *
 * - `cell_to_dof`: for each cell index, the DOF index (-1 if the
 *   cell is not a leaf).
 * - `dof_to_cell`: for each DOF index, the cell index.
 *
 * DOF indices are assigned in cell-array order, which is
 * deterministic (breadth-first refinement order).
 *
 * @param cells        The full octree cell array.
 * @param cell_to_dof  Output mapping (resized to cells.size()).
 * @param dof_to_cell  Output mapping (resized to number of leaves).
 */
inline void assign_dof_indices(
    const std::vector<OctreeCell> &cells,
    std::vector<std::int64_t> &cell_to_dof,
    std::vector<std::size_t> &dof_to_cell) {
    cell_to_dof.assign(cells.size(), -1);
    dof_to_cell.clear();

    /* First pass: count leaves so we can reserve. */
    std::size_t n_leaves = 0;
    for (const auto &c : cells) {
        if (c.is_leaf) {
            ++n_leaves;
        }
    }
    dof_to_cell.reserve(n_leaves);

    /* Second pass: assign indices. */
    std::int64_t next_dof = 0;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].is_leaf) {
            cell_to_dof[i] = next_dof;
            dof_to_cell.push_back(i);
            ++next_dof;
        }
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
 * For degree-1 B-splines, two basis functions overlap if and only
 * if their cell centres are within one cell width in each axis
 * direction.  On a uniform grid this gives a 3^3 = 27 stencil
 * (including self).  On an adaptive grid with 2:1 balance, the
 * stencil can be smaller at depth boundaries.
 *
 * We use a spatial hash to look up neighbors efficiently.  For each
 * DOF, we probe the 27 positions offset by {-1, 0, +1} cell widths
 * along each axis from the cell centre.  Due to 2:1 balance, a
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
    /* Typical stencil has ~27 entries per DOF; pre-allocate
     * conservatively. */
    stencil_neighbors.reserve(n_dofs * 27);

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

        /* Probe all 27 neighbor positions.  Each offset is one
         * cell-width step in fine-grid units (= `span`). */
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
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
                     * Since the stencil is small (<=27), a
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
