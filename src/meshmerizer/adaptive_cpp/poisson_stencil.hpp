/**
 * @file poisson_stencil.hpp
 * @brief Laplacian stencil, screening, and matrix-free operator for
 *        screened Poisson surface reconstruction (Phase 20c).
 *
 * @par References
 * - Kazhdan, M., Bolitho, M. & Hoppe, H. "Poisson Surface
 *   Reconstruction", *Proc. SGP* (2006), Sec. 4.  The Laplacian is
 *   discretised on a B-spline basis using the Galerkin stiffness
 *   matrix L_ij = <grad B_i, grad B_j>. (SGP06)
 * - Kazhdan, M. & Hoppe, H. "Screened Poisson Surface
 *   Reconstruction", *ACM Trans. Graph.* 32(3), Art. 29 (2013),
 *   Sec. 4.3. Screening adds a point-sampled mass term
 *   S_ij = alpha * sum_s 2^{d_s} * B_i(p_s) * B_j(p_s). (ToG13)
 * - Reference implementation: https://github.com/mkazhdan/PoissonRecon
 *   (MIT licence). (PoissonRecon)
 *
 * @par Algorithm
 * The full operator is A = L + S (Laplacian + screening).
 *
 * @par Laplacian stencil (Galerkin stiffness)
 * We use **degree-2 (quadratic) B-splines**. On a uniform grid with
 * cell width h, the 1-D integrals (normalised to unit spacing) are:
 *
 * - Mass/overlap: M(d) = ∫ B₂(t) B₂(t-d) dt
 *     M(0)=11/20, M(±1)=13/60, M(±2)=1/120, else 0
 * - Stiffness:    K(d) = ∫ B₂'(t) B₂'(t-d) dt
 *     K(0)=1,     K(±1)=-1/3,   K(±2)=-1/6,   else 0
 *
 * The 3-D stiffness weight for relative offset (dx,dy,dz) is the sum
 * of contributions from each axis:
 *
 *   L(dx,dy,dz) = h * [ K(dx) M(dy) M(dz)
 *                     + M(dx) K(dy) M(dz)
 *                     + M(dx) M(dy) K(dz) ]
 *
 * Scaling: in each term there is one stiffness integral (1/h after
 * change-of-variables, from the two derivatives) and two mass
 * integrals (h each). Multiplying gives h overall, matching the weak
 * form <grad B_i, grad B_j> integrated over volume. (SGP06)
 *
 * Since degree-2 B-splines have support spanning 3 cells per axis,
 * the stencil support is dx,dy,dz ∈ [-2,2], giving a 5³ = 125-point
 * Galerkin stencil on a uniform grid. Importantly, the face-neighbor
 * coupling L(1,0,0) is nonzero for degree-2, which avoids the
 * even/odd (checkerboard) decoupling of lower-order stiffness.
 * (PoissonRecon)
 *
 * @par Screening term
 * Screening is a point-sampled mass term accumulated per sample:
 *   S_ij += alpha * 2^{d_s} * B_i(p_s) * B_j(p_s). (ToG13)
 *
 * All functions are inline / header-only.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_POISSON_STENCIL_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_POISSON_STENCIL_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "bounding_box.hpp"
#include "octree_cell.hpp"
#include "omp_config.hpp"
#include "poisson_basis.hpp"
#include "poisson_rhs.hpp"
#include "progress_bar.hpp"
#include "vector3d.hpp"

/* ===================================================================
 * Section 1: 3-D Laplacian stencil
 * =================================================================== */

/* poisson_rhs.hpp provides the degree-2 1-D integrals:
 * - mass_integral_1d(d)      -> M(d)
 * - stiffness_integral_1d(d) -> K(d)
 * used by the tensor-product Galerkin stiffness below. */

/**
 * @brief Compute the 3-D Laplacian stencil weight for a given
 *        relative cell offset (dx, dy, dz) and cell width h.
 *
 * This is the **Galerkin stiffness** discretisation with degree-2
 * (quadratic) B-splines, using the tensor-product factorisation of
 * 1-D integrals:
 *
 *   L(dx,dy,dz) = h * [ K(dx) M(dy) M(dz)
 *                     + M(dx) K(dy) M(dz)
 *                     + M(dx) M(dy) K(dz) ]
 *
 * where M and K are the degree-2 integrals (see file header).
 *
 * Support: for degree-2, both M(d) and K(d) are nonzero only for
 * d ∈ {-2,-1,0,+1,+2}, hence the 125-point stencil. (SGP06)
 *
 * Scaling: one stiffness term contributes 1/h after change-of-
 * variables, two mass terms contribute h each, giving an overall
 * factor of h. (SGP06)
 *
 * @note Degree-2 yields nonzero face-neighbor coupling, e.g.
 *   L(1,0,0) = h * [K(1)M(0)M(0) + M(1)K(0)M(0)
 *                + M(1)M(0)K(0)]
 *          ≈ 0.121 * h,
 * which suppresses the checkerboard mode associated with lower-order
 * stiffness discretisations. (PoissonRecon)
 *
 * @param dx  Offset in x (integer: -2..+2).
 * @param dy  Offset in y.
 * @param dz  Offset in z.
 * @param h   Cell width.
 * @return Stiffness integral value (volume-weighted).
 */
inline double laplacian_stencil_weight(int dx, int dy, int dz,
                                        double h) {
    /* Degree-2 B-splines have support over offsets -2..+2 in each
     * axis. For any offset outside this range, the basis functions do
     * not overlap and the Galerkin stiffness is exactly zero. */
    if (dx < -2 || dx > 2 ||
        dy < -2 || dy > 2 ||
        dz < -2 || dz > 2) {
        return 0.0;
    }

    /* Pull degree-2 mass integrals from poisson_rhs.hpp.
     * These are the 1-D overlap integrals M(d). */
    const double mx = mass_integral_1d(dx);
    const double my = mass_integral_1d(dy);
    const double mz = mass_integral_1d(dz);

    /* Degree-2 stiffness integrals K(d) for each axis. */
    const double kx = stiffness_integral_1d(dx);
    const double ky = stiffness_integral_1d(dy);
    const double kz = stiffness_integral_1d(dz);

    /* Tensor-product Galerkin stiffness with the correct h scaling
     * for the weak form <grad B_i, grad B_j>. */
    return h * (kx * my * mz + mx * ky * mz + mx * my * kz);
}

/* ===================================================================
 * Section 2: Screening accumulation
 * =================================================================== */

/**
 * @brief Per-DOF screening data.
 *
 * For each DOF i, we store a list of (neighbor_dof, weight) pairs
 * representing the screening contribution S_ij. For degree-2
 * B-splines, each sample point overlaps at most 3^3 = 27 DOFs, so
 * each sample contributes at most 27*27 = 729 pairs. In practice,
 * the diagonal S_ii often dominates.
 *
 * We store screening in CSR-like format per DOF for efficient
 * matvec.  The diagonal is stored separately for quick access.
 */
struct ScreeningData {
    /** @brief Per-DOF diagonal screening weight. */
    std::vector<double> diagonal;

    /** @brief CSR offsets for off-diagonal entries (size n_dofs+1). */
    std::vector<std::size_t> offsets;

    /** @brief Off-diagonal neighbor DOF indices. */
    std::vector<std::int64_t> neighbors;

    /** @brief Off-diagonal screening weights. */
    std::vector<double> weights;
};

/**
 * @brief Accumulate screening contributions from oriented samples.
 *
 * For each sample point p_s at depth d_s in the octree:
 *   S_ij += alpha * 2^{d_s} * B_i(p_s) * B_j(p_s)
 *
 * where alpha is the global screening weight (ToG13 Sec 4.3
 * recommends alpha ≈ 4 as default).
 *
 * The depth-dependent factor 2^{d_s} ensures that the screening
 * strength scales with resolution: finer cells (higher depth)
 * get stronger screening to preserve detail (ToG13 Sec 4.3).
 *
 * For a uniform grid (all cells at the same depth), 2^d is a
 * constant and the screening reduces to a standard point-sampled
 * mass matrix.
 *
 * @param positions      Sample positions (N points).
 * @param n_samples      Number of samples.
 * @param alpha          Global screening weight.
 * @param hash           Spatial hash of leaf cells.
 * @param cells          Full octree cell array.
 * @param cell_to_dof    Cell-to-DOF mapping.
 * @param dof_to_cell    DOF-to-cell mapping (not currently needed
 *                        for screening depth lookup, but included
 *                        to keep the assembly interface consistent
 *                        with other multi-depth operators).
 * @param n_dofs         Total number of DOFs.
 * @param base_resolution Top-level cells per axis.
 * @param screening      Output screening data.
 */
inline void accumulate_screening(
    const Vector3d *positions,
    std::size_t n_samples,
    double alpha,
    const PoissonLeafHash &hash,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const std::vector<std::size_t> &dof_to_cell,
    std::size_t n_dofs,
    std::uint32_t base_resolution,
    ScreeningData &screening) {
    (void)dof_to_cell;
    /* We accumulate screening into per-DOF neighbor lists.
     *
     * For degree-2, a point overlaps at most 27 DOFs. Across many
     * samples, a single DOF can couple to neighbors within a
     * 5^3 support window, so the number of distinct off-diagonal
     * entries per DOF is bounded by 5^3 - 1 = 124 on a uniform grid.
     * (ToG13, PoissonRecon)
     *
     * We use a small-vector + linear scan approach because neighbor
     * lists are short in typical PoissonRecon-like setups. */

    /* Per-DOF diagonal accumulator. */
    screening.diagonal.assign(n_dofs, 0.0);

    /* Per-DOF off-diagonal accumulators.  We use a map from
     * neighbor DOF index to weight.  For small stencils, a
     * linear scan is fine. */
    struct OffDiagEntry {
        std::int64_t dof;
        double weight;
    };
    std::vector<std::vector<OffDiagEntry>> off_diag(n_dofs);

    ProgressBar progress("Accumulating screening", n_samples);

    std::vector<std::int64_t> dof_indices;
    std::vector<double> bweights;
    dof_indices.reserve(27);
    bweights.reserve(27);

    for (std::size_t s = 0; s < n_samples; ++s) {
        find_overlapping_dofs(positions[s], hash, cells,
                              cell_to_dof, dof_indices,
                              bweights, base_resolution);

        if (dof_indices.empty()) {
            progress.tick();
            continue;
        }

        /* Find the depth of the leaf cell containing this sample.
         *
         * The screening term (ToG13 Sec 4.3) uses a per-sample depth
         * factor 2^{d_s}, where d_s is the depth at which the sample
         * is considered.  In PoissonRecon, this is effectively the
         * depth of the finest cell containing the sample position.
         *
         * The hash's find_leaf_at() probes from finest to coarsest,
         * so it returns the finest leaf that covers the query.
         * (If the point is outside the domain due to numerical
         * roundoff, we fall back to max_depth.) */
        const double sx_q =
            (positions[s].x - hash.domain_min.x) *
            hash.inv_cell_size_x;
        const double sy_q =
            (positions[s].y - hash.domain_min.y) *
            hash.inv_cell_size_y;
        const double sz_q =
            (positions[s].z - hash.domain_min.z) *
            hash.inv_cell_size_z;

        std::size_t sample_ci = SIZE_MAX;
        if (sx_q >= 0.0 && sy_q >= 0.0 && sz_q >= 0.0) {
            sample_ci = hash.find_leaf_at(
                static_cast<std::uint32_t>(sx_q + 0.5),
                static_cast<std::uint32_t>(sy_q + 0.5),
                static_cast<std::uint32_t>(sz_q + 0.5));
        }
        const std::uint32_t sample_depth =
            (sample_ci != SIZE_MAX)
                ? cells[sample_ci].depth
                : hash.max_depth;
        const double depth_factor = static_cast<double>(
            1U << sample_depth);
        const double scale = alpha * depth_factor;

        /* Accumulate S_ij for all 27×27 pairs (i, j) in the
         * overlapping DOF set. */
        const std::size_t nd = dof_indices.size();
        for (std::size_t a = 0; a < nd; ++a) {
            const std::size_t di =
                static_cast<std::size_t>(dof_indices[a]);
            const double wi = bweights[a];

            for (std::size_t b = 0; b < nd; ++b) {
                const std::size_t dj =
                    static_cast<std::size_t>(dof_indices[b]);
                const double wj = bweights[b];
                const double contrib = scale * wi * wj;

                if (di == dj) {
                    /* Diagonal entry. */
                    screening.diagonal[di] += contrib;
                } else {
                    /* Off-diagonal entry.  Find or insert. */
                    auto &entries = off_diag[di];
                    bool found = false;
                    for (auto &e : entries) {
                        if (e.dof ==
                            static_cast<std::int64_t>(dj)) {
                            e.weight += contrib;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        entries.push_back(
                            {static_cast<std::int64_t>(dj),
                             contrib});
                    }
                }
            }
        }

        progress.tick();
    }
    progress.finish();

    /* Compress off-diagonal data into CSR format. */
    screening.offsets.resize(n_dofs + 1);
    screening.offsets[0] = 0;
    std::size_t total_off = 0;
    for (std::size_t i = 0; i < n_dofs; ++i) {
        total_off += off_diag[i].size();
        screening.offsets[i + 1] = total_off;
    }
    screening.neighbors.resize(total_off);
    screening.weights.resize(total_off);
    for (std::size_t i = 0; i < n_dofs; ++i) {
        std::size_t k = screening.offsets[i];
        for (const auto &e : off_diag[i]) {
            screening.neighbors[k] = e.dof;
            screening.weights[k] = e.weight;
            ++k;
        }
    }
}

/* ===================================================================
 * Section 3: Matrix-free operator apply_A
 * =================================================================== */

/**
 * @brief Apply the screened Poisson operator A = L + S to a vector.
 *
 * result[i] = sum_{j in stencil(i)} L_ij * x[j]
 *           + S_ii * x[i]
 *           + sum_{j in screening_neighbors(i)} S_ij * x[j]
 *
 * The Laplacian part uses the analytic 125-point Galerkin stencil
 * computed from degree-2 1-D integrals. The screening part uses the
 * precomputed CSR structure from accumulate_screening().
 *
 * @param x                Input vector (size n_dofs).
 * @param cells            Full octree cell array.
 * @param cell_to_dof      Cell-to-DOF mapping.
 * @param dof_to_cell      DOF-to-cell mapping.
 * @param stencil_offsets  CSR offsets from enumerate_stencils().
 * @param stencil_neighbors CSR neighbors from enumerate_stencils().
 * @param screening        Screening data from accumulate_screening().
 * @param n_dofs           Number of DOFs.
 * @param result           Output vector (size n_dofs).
 */
inline void apply_operator(
    const std::vector<double> &x,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const std::vector<std::size_t> &dof_to_cell,
    const std::vector<std::size_t> &stencil_offsets,
    const std::vector<std::int64_t> &stencil_neighbors,
    const ScreeningData &screening,
    std::size_t n_dofs,
    std::vector<double> &result) {
    result.assign(n_dofs, 0.0);

    for (std::size_t i = 0; i < n_dofs; ++i) {
        const std::size_t ci = dof_to_cell[i];
        const OctreeCell &cell_i = cells[ci];
        const Vector3d center_i = cell_i.bounds.center();
        const double h_i = cell_i.bounds.extent().x;

        double accum = 0.0;

        /* ---- Laplacian part ---- */
        const std::size_t start = stencil_offsets[i];
        const std::size_t end = stencil_offsets[i + 1];

        for (std::size_t k = start; k < end; ++k) {
            const std::int64_t j_dof = stencil_neighbors[k];
            const std::size_t cj = dof_to_cell[
                static_cast<std::size_t>(j_dof)];
            const OctreeCell &cell_j = cells[cj];
            const Vector3d center_j = cell_j.bounds.center();

            /* Relative offset in cell-width units. */
            const double inv_h = 1.0 / h_i;
            const int dx = static_cast<int>(
                std::round((center_j.x - center_i.x) * inv_h));
            const int dy = static_cast<int>(
                std::round((center_j.y - center_i.y) * inv_h));
            const int dz = static_cast<int>(
                std::round((center_j.z - center_i.z) * inv_h));

            /* Skip offsets outside the degree-2 overlap range.
             *
             * Note: even if the neighbor enumeration probes a small
             * set of nearby cells, adaptive depth transitions can
             * yield relative centre offsets of magnitude 2 when
             * measured in the fine cell's width units. */
            if (dx < -2 || dx > 2 ||
                dy < -2 || dy > 2 ||
                dz < -2 || dz > 2) {
                continue;
            }

            const double L_ij =
                laplacian_stencil_weight(dx, dy, dz, h_i);
            accum += L_ij * x[static_cast<std::size_t>(j_dof)];
        }

        /* ---- Screening diagonal ---- */
        accum += screening.diagonal[i] * x[i];

        /* ---- Screening off-diagonal ---- */
        const std::size_t s_start = screening.offsets[i];
        const std::size_t s_end = screening.offsets[i + 1];
        for (std::size_t k = s_start; k < s_end; ++k) {
            const std::size_t j =
                static_cast<std::size_t>(screening.neighbors[k]);
            accum += screening.weights[k] * x[j];
        }

        result[i] = accum;
    }
}

/**
 * @brief Compute the dot product of two vectors.
 *
 * @param a  First vector.
 * @param b  Second vector.
 * @param n  Length.
 * @return dot product.
 */
inline double vec_dot(const std::vector<double> &a,
                      const std::vector<double> &b,
                      std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_POISSON_STENCIL_HPP_
