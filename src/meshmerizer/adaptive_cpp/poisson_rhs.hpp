/**
 * @file poisson_rhs.hpp
 * @brief Vector field splatting and RHS assembly for the screened
 *        Poisson surface reconstruction (Phase 20b).
 *
 * @par References
 * - Kazhdan, M., Bolitho, M. & Hoppe, H. "Poisson Surface
 *   Reconstruction", *Proc. SGP* (2006), Section 3.  The gradient
 *   of a smoothed indicator function equals the smoothed normal
 *   field: grad(chi) ≈ V.  The Poisson equation Laplacian(chi) =
 *   div(V) is obtained by taking the divergence of both sides.
 *   The weak-form RHS is b_i = <V, grad B_i>.
 * - Kazhdan, M. & Hoppe, H. "Screened Poisson Surface
 *   Reconstruction", *ACM Trans. Graph.* 32(3), Art. 29 (2013),
 *   Section 3.  The vector field V is built by splatting each
 *   oriented sample's normal into the B-spline basis, weighted by
 *   the sample's area contribution.
 * - Reference implementation: https://github.com/mkazhdan/PoissonRecon
 *   (MIT licence).
 *
 * @par Algorithm
 * 1. **Splat normals** — For each oriented sample (p_s, n_s):
 *    - Find the leaf cell containing p_s via spatial hash.
 *    - Enumerate the (up to 8) overlapping DOFs whose B-spline
 *      support covers p_s.
 *    - For each such DOF j, accumulate:
 *          V_j += B_j(p_s) * n_s * area_weight_s
 *    where B_j is the trilinear B-spline centred at DOF j and
 *    area_weight_s is a per-sample area correction (default 1/N
 *    for uniformly distributed samples).
 *
 * 2. **Assemble RHS** — The weak-form right-hand side is:
 *        b_i = <V, grad B_i> = integral V(x) · grad B_i(x) dx
 *    Because V is itself expressed in the B-spline basis as
 *    V(x) = sum_j V_j B_j(x), this becomes:
 *        b_i = sum_{j in stencil(i)} V_j · G_{ij}
 *    where G_{ij} = integral B_j(x) grad B_i(x) dx is a
 *    precomputable gradient inner product that depends only on the
 *    relative cell offset (i−j) and cell width h.
 *
 *    For degree-1 B-splines on a uniform grid with cell width h,
 *    the 1-D integrals factor as:
 *
 *    Overlap integral (mass):   M(d) = integral B(t) B(t−d) dt
 *        M(0) = 2/3,  M(±1) = 1/6,  M(else) = 0
 *
 *    Gradient–value integral:   S(d) = integral B'(t) B(t−d) dt
 *        S(0) = 0,  S(+1) = −1/2,  S(−1) = +1/2
 *
 *    The 3-D gradient inner product for offset (dx, dy, dz) is:
 *        G_x = S(dx) * M(dy) * M(dz) / h
 *        G_y = M(dx) * S(dy) * M(dz) / h
 *        G_z = M(dx) * M(dy) * S(dz) / h
 *
 *    (The 1/h comes from the chain rule: d/dx B(x/h) = B'(x/h)/h,
 *     and each integral is over the h-scaled domain so contributes
 *     a factor of h, giving h * (1/h) = 1 for the value-value
 *     part, but the gradient part has an extra 1/h.)
 *
 * All functions are inline / header-only, following the project
 * convention.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_POISSON_RHS_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_POISSON_RHS_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "bounding_box.hpp"
#include "octree_cell.hpp"
#include "omp_config.hpp"
#include "poisson_basis.hpp"
#include "progress_bar.hpp"
#include "vector3d.hpp"

/* ===================================================================
 * Section 1: Precomputed 1-D integrals for degree-1 B-splines
 * =================================================================== */

/**
 * @brief 1-D mass (overlap) integral M(d) for degree-1 B-splines.
 *
 * M(d) = integral_{-inf}^{inf} B(t) B(t - d) dt
 *
 * For the hat function B(t) = max(0, 1 − |t|):
 *   M(0)  = 2/3
 *   M(±1) = 1/6
 *   M(else) = 0
 *
 * @param d Relative offset in cell-width units (integer: -1, 0, +1).
 * @return Integral value.
 */
inline double mass_integral_1d(int d) {
    switch (d) {
        case 0:
            return 2.0 / 3.0;
        case 1:
        case -1:
            return 1.0 / 6.0;
        default:
            return 0.0;
    }
}

/**
 * @brief 1-D gradient-value integral S(d) for degree-1 B-splines.
 *
 * S(d) = integral_{-inf}^{inf} B'(t) B(t - d) dt
 *
 * For the hat function:
 *   S(0)  = 0   (antisymmetric integrand cancels)
 *   S(+1) = -1/2
 *   S(-1) = +1/2
 *
 * Note: S(d) = -S(-d), reflecting the antisymmetry of the
 * derivative operator.
 *
 * @param d Relative offset in cell-width units.
 * @return Integral value.
 */
inline double grad_value_integral_1d(int d) {
    switch (d) {
        case 1:
            return -0.5;
        case -1:
            return 0.5;
        default:
            return 0.0;
    }
}

/* ===================================================================
 * Section 2: 3-D gradient inner product G_{ij}
 * =================================================================== */

/**
 * @brief Compute the 3-D gradient inner product vector G for a
 *        given relative cell offset (dx, dy, dz).
 *
 * G is a 3-component vector where:
 *   G.x = S(dx) * M(dy) * M(dz) / h
 *   G.y = M(dx) * S(dy) * M(dz) / h
 *   G.z = M(dx) * M(dy) * S(dz) / h
 *
 * This is the integral of B_j(x) * grad B_i(x) over the domain,
 * where cell j is offset from cell i by (dx, dy, dz) cell widths.
 *
 * The factor of h^3 from the volume element and the factor of
 * 1/h from the gradient chain rule combine to give h^2 overall.
 * Each 1-D integral is normalised to unit support width, so the
 * actual scaling is h^2 (three factors of h from the volume,
 * minus one factor of h from the gradient).
 *
 * @param dx  Offset in x (integer: -1, 0, +1).
 * @param dy  Offset in y.
 * @param dz  Offset in z.
 * @param h   Cell width.
 * @return Gradient inner product vector.
 */
inline Vector3d gradient_inner_product(int dx, int dy, int dz,
                                       double h) {
    /* Mass integrals for each axis. */
    const double mx = mass_integral_1d(dx);
    const double my = mass_integral_1d(dy);
    const double mz = mass_integral_1d(dz);

    /* Gradient-value integrals for each axis. */
    const double sx = grad_value_integral_1d(dx);
    const double sy = grad_value_integral_1d(dy);
    const double sz = grad_value_integral_1d(dz);

    /* The volume element contributes h^3, and the gradient
     * introduces a 1/h factor (chain rule on the B-spline
     * argument x/h).  Net scaling is h^2.  We express this
     * as h * h to keep the arithmetic transparent. */
    const double scale = h * h;

    return {
        scale * sx * my * mz,
        scale * mx * sy * mz,
        scale * mx * my * sz,
    };
}

/* ===================================================================
 * Section 3: Normal field splatting
 * =================================================================== */

/**
 * @brief Find the DOFs whose B-spline supports overlap a query
 *        point and return their indices with trilinear weights.
 *
 * For a degree-1 B-spline, the support extends one cell width in
 * each direction from the cell centre.  A point can overlap at
 * most 2^3 = 8 DOFs (the 8 cells whose centres bracket the point
 * on each axis).
 *
 * **Approach**: Rather than finding a "containing cell" first
 * (which fails when the hash quantization maps the point to a
 * grid coordinate that doesn't match any cell's min-corner key),
 * we directly compute which cell-aligned min-corner positions
 * could bracket the point.  For each axis, we convert the point
 * to fine-grid coordinates using the same formula as the hash's
 * `quantize()` for min-corners (i.e. floor, not round), then
 * probe the two bracketing positions {floor, floor - span}.
 * All 8 combinations are checked via the hash.
 *
 * @param point       Query point in world space.
 * @param hash        Spatial hash of leaf cells.
 * @param cells       Full octree cell array.
 * @param cell_to_dof Cell-to-DOF mapping.
 * @param dof_indices Output: DOF indices of overlapping cells.
 * @param weights     Output: B-spline weights at the query point.
 * @param base_resolution  Top-level cells per axis.
 */
inline void find_overlapping_dofs(
    const Vector3d &point,
    const PoissonLeafHash &hash,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    std::vector<std::int64_t> &dof_indices,
    std::vector<double> &weights,
    std::uint32_t base_resolution) {
    dof_indices.clear();
    weights.clear();

    /* Compute the fine-grid cell width (same as the leaf cell
     * width at maximum depth).  fine_per_axis = base_resolution
     * * 2^max_depth, so each fine cell spans:
     *   domain_extent / fine_per_axis  in each axis. */
    const std::uint32_t fine_per_axis =
        base_resolution * (1U << hash.max_depth);
    (void)fine_per_axis;  /* Used only for documentation. */

    /* Convert the point to continuous fine-grid coordinates.
     * The hash's inv_cell_size fields map world coords to
     * fine-grid units: t = (point - domain_min) * inv_cell_size.
     * Note: the hash quantize() adds +0.5 and truncates (i.e.
     * rounds), but here we want the raw continuous value. */
    const double tx =
        (point.x - hash.domain_min.x) * hash.inv_cell_size_x;
    const double ty =
        (point.y - hash.domain_min.y) * hash.inv_cell_size_y;
    const double tz =
        (point.z - hash.domain_min.z) * hash.inv_cell_size_z;

    /* For each axis, we need the two cell min-corner indices
     * whose B-spline supports overlap the point.  A cell with
     * min-corner index m has its centre at (m + 0.5) in fine-grid
     * units, and its degree-1 B-spline support covers
     * [m - 0.5, m + 1.5].  A point at fine-grid position t is
     * covered iff |t - (m + 0.5)| < 1, i.e. t - 1.5 < m < t + 0.5.
     * The two integer solutions are floor(t - 0.5) and
     * floor(t - 0.5) + 1. */
    const auto base_idx = [](double t) -> std::int64_t {
        return static_cast<std::int64_t>(std::floor(t - 0.5));
    };

    const std::int64_t ix0 = base_idx(tx);
    const std::int64_t iy0 = base_idx(ty);
    const std::int64_t iz0 = base_idx(tz);

    /* Probe all 8 combinations of {ix0, ix0+1} x {iy0, iy0+1} x
     * {iz0, iz0+1}.  These are the candidate cell min-corner
     * positions in fine-grid coordinates. */
    for (int ox = 0; ox <= 1; ++ox) {
        for (int oy = 0; oy <= 1; ++oy) {
            for (int oz = 0; oz <= 1; ++oz) {
                const std::int64_t px = ix0 + ox;
                const std::int64_t py = iy0 + oy;
                const std::int64_t pz = iz0 + oz;

                if (px < 0 || py < 0 || pz < 0) continue;

                const std::size_t ci = hash.find_leaf_at(
                    static_cast<std::uint32_t>(px),
                    static_cast<std::uint32_t>(py),
                    static_cast<std::uint32_t>(pz));
                if (ci == SIZE_MAX) continue;

                const std::int64_t dof = cell_to_dof[ci];
                if (dof < 0) continue;

                /* Check for duplicates (can happen at depth
                 * boundaries where multiple probes land in the
                 * same coarser cell). */
                bool duplicate = false;
                for (std::int64_t existing : dof_indices) {
                    if (existing == dof) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) continue;

                /* Compute the B-spline weight. */
                const Vector3d center =
                    cells[ci].bounds.center();
                const Vector3d ext = cells[ci].bounds.extent();
                const double w = ext.x;  /* cubic cells */
                const double bval =
                    bspline3d_evaluate(point, center, w);
                if (bval > 0.0) {
                    dof_indices.push_back(dof);
                    weights.push_back(bval);
                }
            }
        }
    }
}

/**
 * @brief Splat oriented point normals into the B-spline basis to
 *        build the smoothed vector field V.
 *
 * For each sample (p_s, n_s) with area weight w_s:
 *   - Find the overlapping DOFs.
 *   - For each DOF j with weight alpha_j = B_j(p_s):
 *       V_j += alpha_j * n_s * w_s
 *
 * The area weight is 1/N for uniform sampling (SGP06 Sec 3.1).
 *
 * @param positions    Sample positions (N points).
 * @param normals      Sample normals (N unit vectors).
 * @param n_samples    Number of samples.
 * @param hash         Spatial hash of leaf cells.
 * @param cells        Full octree cell array.
 * @param cell_to_dof  Cell-to-DOF mapping.
 * @param n_dofs       Total number of DOFs.
 * @param v_field      Output vector field (size n_dofs, zeroed on entry).
 */
inline void splat_normals(
    const Vector3d *positions,
    const Vector3d *normals,
    std::size_t n_samples,
    const PoissonLeafHash &hash,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    std::size_t n_dofs,
    std::uint32_t base_resolution,
    std::vector<Vector3d> &v_field) {
    /* Initialise the vector field to zero. */
    v_field.assign(n_dofs, {0.0, 0.0, 0.0});

    /* Uniform area weight: each sample represents 1/N of the
     * total surface area (SGP06 assumption for uniformly
     * distributed oriented points). */
    const double area_weight = 1.0 / static_cast<double>(
        n_samples > 0 ? n_samples : 1);

    ProgressBar progress("Splatting normals", n_samples);

    /* Per-thread temporary vectors to avoid repeated allocation. */
    /* Note: the splat is over samples (outer loop), and each
     * sample touches at most 8 DOFs.  For thread safety without
     * atomics, we accumulate into per-thread buffers and reduce
     * afterward.  For simplicity in this first implementation,
     * we use a serial loop.  OpenMP parallelism with atomic
     * accumulation or thread-local buffers will be added if
     * profiling shows this is a bottleneck. */
    std::vector<std::int64_t> dof_indices;
    std::vector<double> bweights;
    dof_indices.reserve(8);
    bweights.reserve(8);

    for (std::size_t s = 0; s < n_samples; ++s) {
        find_overlapping_dofs(positions[s], hash, cells,
                              cell_to_dof, dof_indices,
                              bweights, base_resolution);

        for (std::size_t k = 0; k < dof_indices.size(); ++k) {
            const std::size_t dof =
                static_cast<std::size_t>(dof_indices[k]);
            const double w = bweights[k] * area_weight;
            v_field[dof].x += w * normals[s].x;
            v_field[dof].y += w * normals[s].y;
            v_field[dof].z += w * normals[s].z;
        }

        progress.tick();
    }
    progress.finish();
}

/* ===================================================================
 * Section 4: RHS assembly
 * =================================================================== */

/**
 * @brief Compute the RHS vector b for the screened Poisson system.
 *
 * b_i = sum_{j in stencil(i)}  V_j · G_{ij}
 *
 * where G_{ij} is the gradient inner product (integral of
 * B_j * grad B_i) and V_j is the splatted normal field.
 *
 * This function uses the precomputed stencil structure (CSR format)
 * from enumerate_stencils() and computes G on-the-fly from the
 * relative cell offsets.
 *
 * @param v_field           Splatted vector field (size n_dofs).
 * @param cells             Full octree cell array.
 * @param cell_to_dof       Cell-to-DOF mapping.
 * @param dof_to_cell       DOF-to-cell mapping.
 * @param stencil_offsets   CSR offsets (size n_dofs + 1).
 * @param stencil_neighbors CSR neighbor DOF indices.
 * @param n_dofs            Number of DOFs.
 * @param rhs               Output RHS vector (size n_dofs).
 */
inline void compute_rhs(
    const std::vector<Vector3d> &v_field,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const std::vector<std::size_t> &dof_to_cell,
    const std::vector<std::size_t> &stencil_offsets,
    const std::vector<std::int64_t> &stencil_neighbors,
    std::size_t n_dofs,
    std::vector<double> &rhs) {
    rhs.assign(n_dofs, 0.0);

    ProgressBar progress("Assembling RHS", n_dofs);

    for (std::size_t i = 0; i < n_dofs; ++i) {
        const std::size_t ci = dof_to_cell[i];
        const OctreeCell &cell_i = cells[ci];
        const Vector3d center_i = cell_i.bounds.center();
        const double h_i = cell_i.bounds.extent().x;

        /* Quantize cell i's min corner to fine-grid coords so
         * we can compute the relative offset to neighbors. */
        double accum = 0.0;

        const std::size_t start = stencil_offsets[i];
        const std::size_t end = stencil_offsets[i + 1];

        for (std::size_t k = start; k < end; ++k) {
            const std::int64_t j_dof = stencil_neighbors[k];
            const std::size_t cj = dof_to_cell[
                static_cast<std::size_t>(j_dof)];
            const OctreeCell &cell_j = cells[cj];
            const Vector3d center_j = cell_j.bounds.center();

            /* Compute relative offset in cell-width units.
             * On a uniform grid (or between same-depth cells),
             * this gives exact integers {-1, 0, +1}.  For
             * depth-boundary neighbors (2:1 balance), we round
             * to the nearest integer — the stencil integrals
             * are only exact for same-depth pairs but this
             * approximation is standard in adaptive FEM. */
            const double inv_h = 1.0 / h_i;
            const int dx = static_cast<int>(
                std::round((center_j.x - center_i.x) * inv_h));
            const int dy = static_cast<int>(
                std::round((center_j.y - center_i.y) * inv_h));
            const int dz = static_cast<int>(
                std::round((center_j.z - center_i.z) * inv_h));

            /* Skip offsets outside the stencil range.  This can
             * happen at depth boundaries where the rounded offset
             * exceeds ±1 due to different cell sizes. */
            if (dx < -1 || dx > 1 ||
                dy < -1 || dy > 1 ||
                dz < -1 || dz > 1) {
                continue;
            }

            /* G_{ij} is the integral of B_j(x) * grad B_i(x).
             * Note: the gradient is of B_i, not B_j.  The offset
             * from i to j is (dx, dy, dz), so the gradient inner
             * product uses offset (-dx, -dy, -dz) because G is
             * defined as integral B_j grad B_i and we tabulate
             * by the offset of the *value* function relative to
             * the *gradient* function.
             *
             * Actually, let's be precise.  We have:
             *   G_{ij} = integral B_j(x) grad_B_i(x) dx
             * With B_j centred at c_j and B_i centred at c_i,
             * and d = c_j - c_i = (dx, dy, dz) * h:
             *   G_{ij}.x = S(-dx) * M(-dy) * M(-dz) / h * h^3
             *            = h^2 * S(-dx) * M(dy) * M(dz)
             * since M is symmetric and S is antisymmetric.
             * So: G_{ij} = gradient_inner_product(-dx,-dy,-dz,h)
             */
            const Vector3d G = gradient_inner_product(
                -dx, -dy, -dz, h_i);

            /* b_i += V_j · G_{ij} */
            const Vector3d &Vj = v_field[
                static_cast<std::size_t>(j_dof)];
            accum += Vj.dot(G);
        }

        rhs[i] = accum;
        progress.tick();
    }

    progress.finish();
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_POISSON_RHS_HPP_
