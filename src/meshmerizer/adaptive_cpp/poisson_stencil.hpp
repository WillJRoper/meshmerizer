/**
 * @file poisson_stencil.hpp
 * @brief Laplacian stencil, screening, and matrix-free operator
 *        for the screened Poisson surface reconstruction
 *        (Phase 20c).
 *
 * @par References
 * - Kazhdan, M., Bolitho, M. & Hoppe, H. "Poisson Surface
 *   Reconstruction", *Proc. SGP* (2006), Section 4.  The
 *   Laplacian operator is discretised on the B-spline basis
 *   using the stiffness matrix L_ij = <grad B_i, grad B_j>.
 * - Kazhdan, M. & Hoppe, H. "Screened Poisson Surface
 *   Reconstruction", *ACM Trans. Graph.* 32(3), Art. 29 (2013),
 *   Section 4.3.  Screening adds a point-sampled mass term:
 *       S_ij = alpha * sum_s 2^{d_s} * B_i(p_s) * B_j(p_s)
 *   where d_s is the octree depth of the cell containing sample
 *   s, and alpha is the screening weight.
 * - Reference implementation: https://github.com/mkazhdan/PoissonRecon
 *   (MIT licence).
 *
 * @par Algorithm
 * The full operator is A = L + S (Laplacian + screening).
 *
 * **Laplacian stencil (L):**
 * For degree-1 (hat) B-splines on a uniform grid with cell
 * width h, the 1-D integrals are:
 *
 *   Stiffness: K(d) = integral B'(t) B'(t-d) dt
 *     K(0) = 2,  K(±1) = -1,  K(else) = 0
 *
 *   Mass: M(d) = integral B(t) B(t-d) dt
 *     M(0) = 2/3,  M(±1) = 1/6,  M(else) = 0
 *
 * The 3-D stiffness for offset (dx, dy, dz) is the sum of
 * contributions from each axis:
 *   L(dx,dy,dz) = K(dx)*M(dy)*M(dz)
 *               + M(dx)*K(dy)*M(dz)
 *               + M(dx)*M(dy)*K(dz)
 *
 * Scaling: each 1-D integral is over the hat function with
 * support width h.  The substitution t -> x/h gives a factor
 * of h for the mass integral and 1/h for the stiffness integral
 * (two derivatives each contribute 1/h, volume contributes h).
 * In 3-D, each term has one stiffness factor (1/h) and two mass
 * factors (h each), plus the volume h^3, giving:
 *   h^3 * (1/h) * h * h = h^2  ... wait, let's be precise.
 *
 * For a single axis (say x), the stiffness term is:
 *   integral B'_i(x) B'_j(x) dx = (1/h) * K(dx)
 * and the mass terms for the other axes are:
 *   integral B_i(y) B_j(y) dy = h * M(dy)
 *   integral B_i(z) B_j(z) dz = h * M(dz)
 * Combining: (1/h) * K(dx) * h * M(dy) * h * M(dz) = h * K(dx)*M(dy)*M(dz)
 * So the total 3-D scaling is h per term, giving L * h overall.
 *
 * **Screening stencil (S):**
 * For each sample point p_s at depth d_s:
 *   S_ij += alpha * 2^{d_s} * B_i(p_s) * B_j(p_s)
 * This is accumulated during a point-splatting pass similar to
 * the normal field splatting in Phase 20b.
 *
 * **Matrix-free operator:**
 * apply_A(x)[i] = sum_j L_ij * x[j] + sum_j S_ij * x[j]
 * The Laplacian part uses the precomputed 27-point stencil.
 * The screening part uses per-DOF sparse storage.
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
 * Section 1: 1-D stiffness integral for degree-1 B-splines
 * =================================================================== */

/**
 * @brief 1-D stiffness integral K(d) for degree-1 B-splines.
 *
 * K(d) = integral_{-inf}^{inf} B'(t) B'(t - d) dt
 *
 * For the hat function B(t) = max(0, 1 − |t|):
 *   B'(t) = +1 for t in (-1, 0), -1 for t in (0, 1), 0 elsewhere.
 *
 * The convolution of B' with B' yields:
 *   K(0)  = 2
 *   K(±1) = -1
 *   K(else) = 0
 *
 * This is the familiar 1-D finite-difference Laplacian stencil
 * [-1, 2, -1], which arises naturally from integrating the
 * products of hat function derivatives.
 *
 * @param d Relative offset in cell-width units (-1, 0, +1).
 * @return Integral value.
 */
inline double stiffness_integral_1d(int d) {
    switch (d) {
        case 0:
            return 2.0;
        case 1:
        case -1:
            return -1.0;
        default:
            return 0.0;
    }
}

/* ===================================================================
 * Section 2: 3-D Laplacian stencil
 * =================================================================== */

/**
 * @brief Compute the 3-D Laplacian stencil weight for a given
 *        relative cell offset (dx, dy, dz) and cell width h.
 *
 * The Laplacian stencil L(dx, dy, dz) for degree-1 B-splines
 * is the integral of grad B_i · grad B_j over the domain:
 *
 *   L = integral grad B_i(x) · grad B_j(x) dx
 *
 * This factorises as the sum of three axis-aligned terms:
 *
 *   L = (1/h) * K(dx) * h * M(dy) * h * M(dz)
 *     + h * M(dx) * (1/h) * K(dy) * h * M(dz)
 *     + h * M(dx) * h * M(dy) * (1/h) * K(dz)
 *
 *   = h * [ K(dx)*M(dy)*M(dz) + M(dx)*K(dy)*M(dz)
 *         + M(dx)*M(dy)*K(dz) ]
 *
 * @param dx  Offset in x (integer: -1, 0, +1).
 * @param dy  Offset in y.
 * @param dz  Offset in z.
 * @param h   Cell width.
 * @return Stiffness integral value.
 */
inline double laplacian_stencil_weight(int dx, int dy, int dz,
                                       double h) {
    const double kx = stiffness_integral_1d(dx);
    const double ky = stiffness_integral_1d(dy);
    const double kz = stiffness_integral_1d(dz);

    const double mx = mass_integral_1d(dx);
    const double my = mass_integral_1d(dy);
    const double mz = mass_integral_1d(dz);

    return h * (kx * my * mz + mx * ky * mz + mx * my * kz);
}

/* ===================================================================
 * Section 3: Screening accumulation
 * =================================================================== */

/**
 * @brief Per-DOF screening data.
 *
 * For each DOF i, we store a list of (neighbor_dof, weight) pairs
 * representing the screening contribution S_ij.  For degree-1
 * B-splines, each sample point overlaps at most 8 DOFs, so each
 * sample contributes at most 8*8 = 64 pairs.  In practice, the
 * diagonal S_ii dominates.
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
    std::size_t n_dofs,
    std::uint32_t base_resolution,
    ScreeningData &screening) {
    /* We first accumulate into a dense per-DOF map of
     * (neighbor -> weight).  For large grids, a hash map per DOF
     * would be more memory-efficient, but for simplicity we use
     * a flat vector of maps represented as sorted pairs.
     *
     * Actually, since the number of off-diagonal entries per DOF
     * is bounded by 26 (3^3 - 1), we use a simpler approach:
     * accumulate into a dense (n_dofs x 27) array indexed by
     * the 27 stencil offsets, then compress to CSR. */

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
    dof_indices.reserve(8);
    bweights.reserve(8);

    for (std::size_t s = 0; s < n_samples; ++s) {
        find_overlapping_dofs(positions[s], hash, cells,
                              cell_to_dof, dof_indices,
                              bweights, base_resolution);

        if (dof_indices.empty()) {
            progress.tick();
            continue;
        }

        /* Find the depth of the containing cell.  We use the
         * first overlapping DOF's cell depth as a proxy (on a
         * uniform grid, all cells have the same depth). */
        const std::size_t first_ci = static_cast<std::size_t>(
            dof_indices[0]);
        /* Actually we need the cell index, not the DOF index.
         * The DOF-to-cell mapping isn't available here, so we
         * look up from the hash instead.  For the first DOF
         * found, the cell depth is cells[ci].depth where ci
         * is the cell containing the point.  Since we have the
         * dof_indices and the B-spline weights, the depth of
         * the nearest cell (highest weight) is a good proxy.
         * On a uniform grid all depths are equal. */
        /* Simpler: find the cell index for the first DOF by
         * searching cell_to_dof.  But that's O(N).  Instead,
         * we'll use a different approach: re-query the hash for
         * the point itself. */
        /* Actually the most robust approach: for each DOF in
         * dof_indices, look up the cell via a reverse search.
         * But we don't have dof_to_cell here.  Let's just
         * iterate the cells array to find the cell for the
         * first DOF.  This is O(n_cells) per sample, which is
         * too slow.
         *
         * Better: since all cells at max_depth have depth =
         * max_depth, and uniform grids have constant depth,
         * we use max_depth as the depth for screening.  On
         * adaptive grids, a more precise lookup would be needed,
         * but for Phase 20c we target uniform grids. */
        const double depth_factor = static_cast<double>(
            1U << hash.max_depth);
        const double scale = alpha * depth_factor;

        /* Accumulate S_ij for all pairs (i, j) in the
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
 * Section 4: Matrix-free operator apply_A
 * =================================================================== */

/**
 * @brief Apply the screened Poisson operator A = L + S to a vector.
 *
 * result[i] = sum_{j in stencil(i)} L_ij * x[j]
 *           + S_ii * x[i]
 *           + sum_{j in screening_neighbors(i)} S_ij * x[j]
 *
 * The Laplacian part uses the analytic 27-point stencil computed
 * from the 1-D integrals.  The screening part uses the precomputed
 * CSR structure from accumulate_screening().
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

            /* Skip offsets outside the stencil range. */
            if (dx < -1 || dx > 1 ||
                dy < -1 || dy > 1 ||
                dz < -1 || dz > 1) {
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
