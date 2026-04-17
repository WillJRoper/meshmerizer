/**
 * @file poisson_solver.hpp
 * @brief Conjugate Gradient solver with diagonal (Jacobi)
 *        preconditioning for the screened Poisson system
 *        (Phase 20d).
 *
 * @par References
 * - Kazhdan, M. & Hoppe, H. "Screened Poisson Surface
 *   Reconstruction", *ACM Trans. Graph.* 32(3), Art. 29 (2013),
 *   Section 4.  The cascadic multigrid solver is recommended for
 *   adaptive octrees.  For uniform grids (all cells at the same
 *   depth), the system reduces to a single-level solve.  We use
 *   preconditioned Conjugate Gradients (PCG) with Jacobi (diagonal)
 *   preconditioning, which is robust and efficient for the SPD
 *   screened Poisson operator.
 * - Kazhdan, M., Bolitho, M. & Hoppe, H. "Poisson Surface
 *   Reconstruction", *Proc. SGP* (2006), Section 4.
 * - Reference implementation: https://github.com/mkazhdan/PoissonRecon
 *   (MIT licence).
 *
 * @par Algorithm
 * **Preconditioned Conjugate Gradients (PCG)**:
 *
 * Given SPD operator A and RHS b, solve A x = b:
 *   1. r = b - A x_0       (initial residual)
 *   2. z = M^{-1} r        (precondition: M = diag(A))
 *   3. p = z               (initial search direction)
 *   4. For k = 1, 2, ..., max_iters:
 *      a. q = A p
 *      b. alpha = (r · z) / (p · q)
 *      c. x += alpha * p
 *      d. r -= alpha * q
 *      e. Check |r| < tol * |b|
 *      f. z_new = M^{-1} r
 *      g. beta = (r · z_new) / (r · z)
 *      h. p = z_new + beta * p
 *      i. z = z_new
 *
 * The Jacobi preconditioner M = diag(A) is cheap to compute
 * (just the Laplacian diagonal + screening diagonal) and
 * typically reduces iteration count by 2-3x vs unpreconditioned CG.
 *
 * **Gauss-Seidel smoother** is also provided for potential
 * multigrid use and as an alternative smoother.
 *
 * All functions are inline / header-only.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_POISSON_SOLVER_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_POISSON_SOLVER_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "bounding_box.hpp"
#include "octree_cell.hpp"
#include "omp_config.hpp"
#include "poisson_basis.hpp"
#include "poisson_stencil.hpp"
#include "progress_bar.hpp"
#include "vector3d.hpp"

/* ===================================================================
 * Section 1: Vector arithmetic helpers
 * =================================================================== */

/**
 * @brief Compute r = b - A*x (residual).
 *
 * @param Ax    Pre-computed A*x.
 * @param b     Right-hand side.
 * @param r     Output residual.
 * @param n     Vector length.
 */
inline void compute_residual(
    const std::vector<double> &Ax,
    const std::vector<double> &b,
    std::vector<double> &r,
    std::size_t n) {
    r.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = b[i] - Ax[i];
    }
}

/**
 * @brief x += alpha * p (axpy operation).
 */
inline void vec_axpy(std::vector<double> &x,
                     double alpha,
                     const std::vector<double> &p,
                     std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
    }
}

/**
 * @brief Compute the L2 norm of a vector.
 */
inline double vec_norm(const std::vector<double> &v,
                       std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += v[i] * v[i];
    }
    return std::sqrt(s);
}

/* ===================================================================
 * Section 2: Jacobi (diagonal) preconditioner
 * =================================================================== */

/**
 * @brief Extract the diagonal of the operator A = L + S.
 *
 * The diagonal of the Galerkin Laplacian stencil at DOF i is
 * L(0,0,0,h_i) = h_i * 3 * K(0) * M(0) * M(0)
 * For degree-2 B-splines:
 *   = h_i * 3 * 1 * (11/20)^2 = h_i * 363/400 ≈ 0.9075 h_i
 *
 * The screening diagonal is stored in screening.diagonal[i].
 *
 * @param cells       Octree cells.
 * @param dof_to_cell DOF-to-cell mapping.
 * @param screening   Screening data.
 * @param n_dofs      Number of DOFs.
 * @param diag        Output diagonal (size n_dofs).
 */
inline void extract_diagonal(
    const std::vector<OctreeCell> &cells,
    const std::vector<std::size_t> &dof_to_cell,
    const ScreeningData &screening,
    std::size_t n_dofs,
    std::vector<double> &diag) {
    diag.resize(n_dofs);
    for (std::size_t i = 0; i < n_dofs; ++i) {
        const std::size_t ci = dof_to_cell[i];
        const double h = cells[ci].bounds.extent().x;
        /* L(0,0,0,h) = h * 3 * K(0)*M(0)*M(0)
         * For degree-2: h * 3 * (2/3) * (11/20)^2
         *             = h * 363/400 ≈ 0.9075h */
        diag[i] = laplacian_stencil_weight(0, 0, 0, h)
                  + screening.diagonal[i];
    }
}

/**
 * @brief Apply Jacobi preconditioner: z = M^{-1} r.
 *
 * M^{-1} is the inverse diagonal of A.  For safety, we clamp
 * the diagonal to a minimum value to avoid division by zero
 * (shouldn't happen for the screened Poisson operator, but
 * defensive coding).
 *
 * @param r     Input residual.
 * @param diag  Diagonal of A.
 * @param z     Output preconditioned residual.
 * @param n     Vector length.
 */
inline void apply_jacobi(
    const std::vector<double> &r,
    const std::vector<double> &diag,
    std::vector<double> &z,
    std::size_t n) {
    z.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double d = diag[i];
        /* Clamp to avoid division by zero.  The diagonal of
         * the screened Poisson operator should always be
         * positive (L diagonal is positive, screening is
         * non-negative), but we guard defensively. */
        z[i] = (std::abs(d) > 1e-30) ? r[i] / d : 0.0;
    }
}

/* ===================================================================
 * Section 3: Gauss-Seidel smoother
 * =================================================================== */

/**
 * @brief Perform one forward Gauss-Seidel sweep.
 *
 * For each DOF i (in order 0, 1, ..., n-1):
 *   x[i] = (b[i] - sum_{j != i} A_ij * x[j]) / A_ii
 *
 * This is a serial sweep.  For parallel multi-colour GS,
 * DOFs would be partitioned into independent colour sets.
 * For Phase 20d we use serial GS within the CG solver as
 * an optional smoother; parallelism is future work.
 *
 * @param x                 Solution vector (updated in-place).
 * @param b                 Right-hand side.
 * @param cells             Octree cells.
 * @param cell_to_dof       Cell-to-DOF mapping.
 * @param dof_to_cell       DOF-to-cell mapping.
 * @param stencil_offsets   CSR offsets.
 * @param stencil_neighbors CSR neighbors.
 * @param stencil_depth_deltas CSR depth deltas (0,-1,+1).
 * @param screening         Screening data.
 * @param diag              Diagonal of A.
 * @param n_dofs            Number of DOFs.
 */
inline void gauss_seidel_sweep(
    std::vector<double> &x,
    const std::vector<double> &b,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const std::vector<std::size_t> &dof_to_cell,
    const std::vector<std::size_t> &stencil_offsets,
    const std::vector<std::int64_t> &stencil_neighbors,
    const std::vector<int> &stencil_depth_deltas,
    const ScreeningData &screening,
    const std::vector<double> &diag,
    std::size_t n_dofs) {

    for (std::size_t i = 0; i < n_dofs; ++i) {
        const std::size_t ci = dof_to_cell[i];
        const OctreeCell &cell_i = cells[ci];
        const Vector3d center_i = cell_i.bounds.center();
        const double h_i = cell_i.bounds.extent().x;

        double off_diag_sum = 0.0;

        /* Laplacian off-diagonal. */
        const std::size_t start = stencil_offsets[i];
        const std::size_t end = stencil_offsets[i + 1];

        for (std::size_t k = start; k < end; ++k) {
            const std::int64_t j_dof = stencil_neighbors[k];
            const std::size_t j =
                static_cast<std::size_t>(j_dof);
            if (j == i) continue;  /* skip diagonal */

            const std::size_t cj = dof_to_cell[j];
            const OctreeCell &cell_j = cells[cj];
            const Vector3d center_j = cell_j.bounds.center();

            /* --------------------------------------------------------
             * Adaptive Laplacian off-diagonal (same-depth CC vs
             * cross-depth PC).
             *
             * IMPORTANT: This must mirror apply_operator() exactly.
             * If the smoother uses a different A_ij, Gauss-Seidel can
             * diverge or destroy PCG convergence.
             *
             * BUGFIX (Phase 21e): same-depth overlap radius is ±2 for
             * degree-2 B-splines (5^3 stencil), not ±1.
             * -------------------------------------------------------- */
            const int depth_delta = stencil_depth_deltas[k];

            double L_ij;
            if (depth_delta == 0) {
                /* Same-depth: integer offsets in h_i units (±2). */
                const double inv_h = 1.0 / h_i;
                const int dx = static_cast<int>(
                    std::round(
                        (center_j.x - center_i.x) * inv_h));
                const int dy = static_cast<int>(
                    std::round(
                        (center_j.y - center_i.y) * inv_h));
                const int dz = static_cast<int>(
                    std::round(
                        (center_j.z - center_i.z) * inv_h));
                if (dx < -2 || dx > 2 ||
                    dy < -2 || dy > 2 ||
                    dz < -2 || dz > 2) {
                    continue;
                }
                L_ij = laplacian_stencil_weight(dx, dy, dz, h_i);
            } else if (depth_delta == -1) {
                /* Neighbour j is coarser (parent of i).
                 *
                 * The PC Laplacian table is indexed in child-width
                 * units, and physical scaling is 1/(2*h_child).
                 * (ToG13 Sec 4; PoissonRecon BSplineData.inl) */
                const double inv_h = 1.0 / h_i;
                const int dx = static_cast<int>(
                    std::round(
                        (center_j.x - center_i.x) * inv_h));
                const int dy = static_cast<int>(
                    std::round(
                        (center_j.y - center_i.y) * inv_h));
                const int dz = static_cast<int>(
                    std::round(
                        (center_j.z - center_i.z) * inv_h));
                if (dx < -4 || dx > 4 ||
                    dy < -4 || dy > 4 ||
                    dz < -4 || dz > 4) {
                    continue;
                }
                const double raw = pc_laplacian_stencil_weight(
                    -dx, -dy, -dz);
                L_ij = raw / (2.0 * h_i);
            } else {
                /* depth_delta == +1: neighbour j is finer (child of i).
                 * Scale is 1/(2*h_child) with offsets in child units.
                 * (ToG13 Sec 4) */
                const double h_j = cell_j.bounds.extent().x;
                const double inv_hj = 1.0 / h_j;
                const int dx = static_cast<int>(
                    std::round(
                        (center_j.x - center_i.x) * inv_hj));
                const int dy = static_cast<int>(
                    std::round(
                        (center_j.y - center_i.y) * inv_hj));
                const int dz = static_cast<int>(
                    std::round(
                        (center_j.z - center_i.z) * inv_hj));
                if (dx < -4 || dx > 4 ||
                    dy < -4 || dy > 4 ||
                    dz < -4 || dz > 4) {
                    continue;
                }
                const double raw = pc_laplacian_stencil_weight(
                    dx, dy, dz);
                L_ij = raw / (2.0 * h_j);
            }

            off_diag_sum += L_ij * x[j];
        }

        /* Screening off-diagonal. */
        const std::size_t s_start = screening.offsets[i];
        const std::size_t s_end = screening.offsets[i + 1];
        for (std::size_t k = s_start; k < s_end; ++k) {
            off_diag_sum +=
                screening.weights[k]
                * x[static_cast<std::size_t>(
                      screening.neighbors[k])];
        }

        /* Update x[i]. */
        const double d = diag[i];
        if (std::abs(d) > 1e-30) {
            x[i] = (b[i] - off_diag_sum) / d;
        }
    }
}

/* ===================================================================
 * Section 4: Preconditioned Conjugate Gradient solver
 * =================================================================== */

/**
 * @brief Result structure for the PCG solver.
 */
struct SolverResult {
    /** @brief Number of iterations performed. */
    std::size_t iterations;
    /** @brief Final relative residual norm |r| / |b|. */
    double residual_norm;
    /** @brief True if converged within tolerance. */
    bool converged;
};

/**
 * @brief Solve the screened Poisson system A x = b using
 *        Preconditioned Conjugate Gradients with Jacobi
 *        preconditioning.
 *
 * The operator A = L + S is applied matrix-free using
 * apply_operator() from poisson_stencil.hpp.
 *
 * @param b                 Right-hand side (size n_dofs).
 * @param cells             Octree cells.
 * @param cell_to_dof       Cell-to-DOF mapping.
 * @param dof_to_cell       DOF-to-cell mapping.
 * @param stencil_offsets   CSR offsets.
 * @param stencil_neighbors CSR neighbors.
 * @param stencil_depth_deltas CSR depth deltas (0,-1,+1).
 * @param screening         Screening data.
 * @param n_dofs            Number of DOFs.
 * @param max_iters         Maximum CG iterations.
 * @param tol               Relative residual tolerance.
 * @param x                 Solution vector (initial guess on
 *                          entry, solution on exit).
 * @return Solver result with convergence info.
 */
inline SolverResult solve_pcg(
    const std::vector<double> &b,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const std::vector<std::size_t> &dof_to_cell,
    const std::vector<std::size_t> &stencil_offsets,
    const std::vector<std::int64_t> &stencil_neighbors,
    const std::vector<int> &stencil_depth_deltas,
    const ScreeningData &screening,
    std::size_t n_dofs,
    std::size_t max_iters,
    double tol,
    std::vector<double> &x) {

    SolverResult result = {0, 0.0, false};

    /* Extract diagonal for Jacobi preconditioner. */
    std::vector<double> diag;
    extract_diagonal(cells, dof_to_cell, screening,
                     n_dofs, diag);

    /* Compute initial residual r = b - A*x. */
    std::vector<double> Ax;
    apply_operator(x, cells, cell_to_dof, dof_to_cell,
                   stencil_offsets, stencil_neighbors,
                   stencil_depth_deltas,
                   screening, n_dofs, Ax);

    std::vector<double> r;
    compute_residual(Ax, b, r, n_dofs);

    const double b_norm = vec_norm(b, n_dofs);
    const double abs_tol =
        (b_norm > 1e-30) ? tol * b_norm : tol;

    double r_norm = vec_norm(r, n_dofs);
    if (r_norm < abs_tol) {
        result.residual_norm = r_norm / (b_norm + 1e-30);
        result.converged = true;
        return result;
    }

    /* Precondition: z = M^{-1} r. */
    std::vector<double> z;
    apply_jacobi(r, diag, z, n_dofs);

    /* Search direction p = z. */
    std::vector<double> p = z;

    /* r · z. */
    double rz = vec_dot(r, z, n_dofs);

    /* Temporary vector for A*p. */
    std::vector<double> q;

    ProgressBar progress("PCG solve", max_iters);

    for (std::size_t iter = 0; iter < max_iters; ++iter) {
        /* q = A * p. */
        apply_operator(p, cells, cell_to_dof, dof_to_cell,
                       stencil_offsets, stencil_neighbors,
                       stencil_depth_deltas,
                       screening, n_dofs, q);

        /* alpha = (r · z) / (p · q). */
        const double pq = vec_dot(p, q, n_dofs);
        if (std::abs(pq) < 1e-30) break;  /* breakdown */
        const double alpha = rz / pq;

        /* x += alpha * p. */
        vec_axpy(x, alpha, p, n_dofs);

        /* r -= alpha * q. */
        vec_axpy(r, -alpha, q, n_dofs);

        /* Check convergence. */
        r_norm = vec_norm(r, n_dofs);
        result.iterations = iter + 1;
        result.residual_norm = r_norm / (b_norm + 1e-30);

        progress.tick();

        if (r_norm < abs_tol) {
            result.converged = true;
            break;
        }

        /* z_new = M^{-1} r. */
        apply_jacobi(r, diag, z, n_dofs);

        /* beta = (r · z_new) / (r · z_old). */
        const double rz_new = vec_dot(r, z, n_dofs);
        const double beta =
            (std::abs(rz) > 1e-30) ? rz_new / rz : 0.0;

        /* p = z + beta * p. */
        for (std::size_t i = 0; i < n_dofs; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        rz = rz_new;
    }

    progress.finish();
    return result;
}

/* ===================================================================
 * Section 5: Cascadic depth-by-depth solver (ToG13 Algorithm 1)
 * =================================================================== */

/**
 * @brief Apply the same-depth-only operator block A_dd.
 *
 * This helper applies the restriction of the full screened Poisson
 * operator A = L + S to a single octree depth d (a contiguous DOF
 * range [dof_begin, dof_end)).  The resulting local mat-vec is:
 *
 *   (A_dd x_d)[i] = sum_{j in depth d} L_ij x_d[j]
 *                + S_ii x_d[i]
 *                + sum_{j in depth d} S_ij x_d[j]
 *
 * where we keep:
 * - Laplacian stencil entries with depth_delta == 0 (same depth), and
 * - screening entries whose neighbor DOF j is also in [dof_begin, dof_end).
 *
 * Cross-depth couplings (parent-child Laplacian and screening between
 * depths) are *intentionally omitted* here; in the cascadic solver
 * (Kazhdan & Hoppe, ToG13, Algorithm 1) they are handled by moving
 * already-solved coarser contributions to the right-hand side.
 *
 * @note This must mirror apply_operator() for the retained entries.
 *
 * @par References
 * - Kazhdan, M. & Hoppe, H. (2013). Screened Poisson Surface
 *   Reconstruction. ACM Trans. Graph. (ToG13), Algorithm 1.
 * - Kazhdan, M., Bolitho, M. & Hoppe, H. (2006). Poisson Surface
 *   Reconstruction. SGP06.
 * - PoissonRecon reference implementation:
 *   https://github.com/mkazhdan/PoissonRecon
 */
inline void apply_operator_depth(
    const std::vector<double> &x_local,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const std::vector<std::size_t> &dof_to_cell,
    const std::vector<std::size_t> &stencil_offsets,
    const std::vector<std::int64_t> &stencil_neighbors,
    const std::vector<int> &stencil_depth_deltas,
    const ScreeningData &screening,
    std::size_t dof_begin,
    std::size_t dof_end,
    std::vector<double> &result_local) {

    /* cell_to_dof is part of the standard operator signature even
     * though the current mat-vec only needs dof_to_cell. */
    (void)cell_to_dof;

    const std::size_t n_local = dof_end - dof_begin;
    result_local.assign(n_local, 0.0);

    for (std::size_t i_global = dof_begin; i_global < dof_end; ++i_global) {
        const std::size_t i_local = i_global - dof_begin;

        const std::size_t ci = dof_to_cell[i_global];
        const OctreeCell &cell_i = cells[ci];
        const Vector3d center_i = cell_i.bounds.center();
        const double h_i = cell_i.bounds.extent().x;

        double accum = 0.0;

        /* ---- Laplacian part (same depth only) ----
         *
         * We keep only stencil entries with depth_delta == 0 and with
         * the neighbour DOF also inside the local range.
         *
         * This is the A_dd block in ToG13's cascadic solver: it is the
         * operator induced by basis functions at a fixed depth.
         */
        const std::size_t start = stencil_offsets[i_global];
        const std::size_t end = stencil_offsets[i_global + 1];

        for (std::size_t k = start; k < end; ++k) {
            if (stencil_depth_deltas[k] != 0) {
                /* Parent-child coupling is not part of A_dd. */
                continue;
            }

            const std::int64_t j_dof = stencil_neighbors[k];
            if (j_dof < 0) continue;
            const std::size_t j_global = static_cast<std::size_t>(j_dof);

            /* Restrict to same-depth unknowns. */
            if (j_global < dof_begin || j_global >= dof_end) {
                continue;
            }
            const std::size_t j_local = j_global - dof_begin;

            const std::size_t cj = dof_to_cell[j_global];
            const OctreeCell &cell_j = cells[cj];
            const Vector3d center_j = cell_j.bounds.center();

            /* Same-depth overlap for degree-2 B-splines is ±2 cell
             * widths per axis (5^3 stencil). */
            const double inv_h = 1.0 / h_i;
            const int dx = static_cast<int>(
                std::round((center_j.x - center_i.x) * inv_h));
            const int dy = static_cast<int>(
                std::round((center_j.y - center_i.y) * inv_h));
            const int dz = static_cast<int>(
                std::round((center_j.z - center_i.z) * inv_h));
            if (dx < -2 || dx > 2 || dy < -2 || dy > 2 || dz < -2 ||
                dz > 2) {
                continue;
            }

            const double L_ij =
                laplacian_stencil_weight(dx, dy, dz, h_i);
            accum += L_ij * x_local[j_local];
        }

        /* ---- Screening diagonal ----
         *
         * The screening diagonal S_ii is depth-local by definition.
         */
        accum += screening.diagonal[i_global] * x_local[i_local];

        /* ---- Screening off-diagonal (restricted) ----
         *
         * The screening neighbor list can include cross-depth pairs.
         * For A_dd, we retain only neighbours also in the local range.
         */
        const std::size_t s_start = screening.offsets[i_global];
        const std::size_t s_end = screening.offsets[i_global + 1];
        for (std::size_t k = s_start; k < s_end; ++k) {
            const std::int64_t j_dof = screening.neighbors[k];
            if (j_dof < 0) continue;
            const std::size_t j_global = static_cast<std::size_t>(j_dof);
            if (j_global < dof_begin || j_global >= dof_end) {
                continue;
            }
            const std::size_t j_local = j_global - dof_begin;
            accum += screening.weights[k] * x_local[j_local];
        }

        result_local[i_local] = accum;
    }
}

/**
 * @brief Cascadic depth-by-depth solve for adaptive octrees.
 *
 * This implements the depth-by-depth cascadic solver recommended in
 * Kazhdan & Hoppe (ToG13, Algorithm 1) for adaptive Poisson surface
 * reconstruction.
 *
 * The key idea is to avoid solving the full multi-depth system in one
 * go.  Instead, we iterate depths d = 0..max_depth and at each depth:
 *
 *  1. Treat already-solved coarser unknowns (depth < d) as fixed
 *     constraints, moving their contribution A_{i,<d} x_{<d} to the
 *     right-hand side.
 *  2. Solve the restricted same-depth system A_dd x_d = rhs_eff using
 *     PCG, where A_dd includes:
 *       - same-depth Laplacian stencil (depth_delta == 0)
 *       - screening diagonal and same-depth screening off-diagonal.
 *
 * This produces a good initialisation for finer levels and matches the
 * cascadic scheme used in PoissonRecon for adaptive octrees.
 *
 * @par References
 * - Kazhdan, M. & Hoppe, H. (2013). Screened Poisson Surface
 *   Reconstruction. ACM Trans. Graph. (ToG13), Algorithm 1.
 * - PoissonRecon reference implementation:
 *   https://github.com/mkazhdan/PoissonRecon
 */
inline SolverResult solve_cascadic(
    const std::vector<double> &b,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const std::vector<std::size_t> &dof_to_cell,
    const std::vector<std::size_t> &stencil_offsets,
    const std::vector<std::int64_t> &stencil_neighbors,
    const std::vector<int> &stencil_depth_deltas,
    const ScreeningData &screening,
    const std::vector<std::int64_t> &depth_dof_start,
    std::uint32_t max_depth,
    std::size_t n_dofs,
    std::size_t max_iters_per_depth,
    double tol,
    std::vector<double> &x) {

    SolverResult result = {0, 0.0, true};

    if (n_dofs == 0) {
        /* Degenerate case: empty system. */
        x.clear();
        result.residual_norm = 0.0;
        result.converged = true;
        return result;
    }

    /* Ensure x has the correct size. We do not overwrite the existing
     * contents, so callers may provide an initial guess (typically the
     * cascadic scheme uses x=0 at start). */
    if (x.size() != n_dofs) {
        x.assign(n_dofs, 0.0);
    }

    /* --------------------------------------------------------------
     * Depth-by-depth loop.
     *
     * depth_dof_start has size max_depth+2, and provides contiguous
     * DOF ranges by depth:
     *   [depth_dof_start[d], depth_dof_start[d+1])
     *
     * This ordering is required for the cascadic algorithm.
     * -------------------------------------------------------------- */
    for (std::uint32_t d = 0U; d <= max_depth; ++d) {
        const std::size_t dof_begin = static_cast<std::size_t>(
            depth_dof_start[static_cast<std::size_t>(d)]);
        const std::size_t dof_end = static_cast<std::size_t>(
            depth_dof_start[static_cast<std::size_t>(d + 1U)]);

        if (dof_begin >= dof_end) {
            /* No unknowns at this depth. */
            continue;
        }

        const std::size_t n_local = dof_end - dof_begin;

        /* Depth-0 typically has only a handful of DOFs (1-8 in many
         * reconstructions).  We can afford a tighter tolerance here to
         * reduce error propagation to finer levels (ToG13's cascadic
         * scheme relies on accurate coarse solutions). */
        const double tol_d = (d == 0U && tol > 1e-12) ? 1e-12 : tol;

        /* ----------------------------------------------------------
         * Step 1: Compute coarser-depth constraint term.
         *
         * For each i at depth d, compute:
         *   constraint[i] = sum_{j depth < d} A_ij * x[j]
         *
         * This uses:
         * - cross-depth Laplacian entries (parent-child couplings), and
         * - screening off-diagonal entries to already-solved DOFs.
         *
         * These contributions are then subtracted from b to form the
         * effective RHS for the depth-d restricted solve.
         * ---------------------------------------------------------- */
        std::vector<double> rhs_eff(n_local, 0.0);
        for (std::size_t i_global = dof_begin; i_global < dof_end;
             ++i_global) {
            const std::size_t i_local = i_global - dof_begin;

            const std::size_t ci = dof_to_cell[i_global];
            const OctreeCell &cell_i = cells[ci];
            const Vector3d center_i = cell_i.bounds.center();
            const double h_i = cell_i.bounds.extent().x;

            double constraint = 0.0;

            /* ---- Cross-depth Laplacian couplings ----
             *
             * We iterate the full CSR stencil for i, but keep only
             * entries with depth_delta != 0 and with neighbour depth
             * strictly smaller than d (already solved).
             */
            const std::size_t start = stencil_offsets[i_global];
            const std::size_t end = stencil_offsets[i_global + 1];

            for (std::size_t k = start; k < end; ++k) {
                if (stencil_depth_deltas[k] == 0) {
                    /* Same-depth terms belong to A_dd. */
                    continue;
                }

                const std::int64_t j_dof = stencil_neighbors[k];
                if (j_dof < 0) continue;
                const std::size_t j_global =
                    static_cast<std::size_t>(j_dof);

                const std::size_t cj = dof_to_cell[j_global];
                const OctreeCell &cell_j = cells[cj];

                /* Only move contributions from already-solved DOFs. */
                if (cell_j.depth >= d) {
                    continue;
                }

                const Vector3d center_j = cell_j.bounds.center();

                /* The Laplacian coupling weight must match
                 * apply_operator() exactly for symmetry/consistency.
                 * (ToG13; PoissonRecon BSplineData.inl) */
                const int depth_delta = stencil_depth_deltas[k];

                double L_ij;
                if (depth_delta == -1) {
                    /* Neighbour j is coarser (parent of i).
                     *
                     * Offset is measured in child-width units:
                     *   (center_j - center_i) / h_i
                     * and the table expects parent->child, so we flip
                     * sign (see apply_operator()). */
                    const double inv_h = 1.0 / h_i;
                    const int dx = static_cast<int>(
                        std::round((center_j.x - center_i.x) * inv_h));
                    const int dy = static_cast<int>(
                        std::round((center_j.y - center_i.y) * inv_h));
                    const int dz = static_cast<int>(
                        std::round((center_j.z - center_i.z) * inv_h));
                    if (dx < -4 || dx > 4 || dy < -4 || dy > 4 ||
                        dz < -4 || dz > 4) {
                        continue;
                    }
                    const double raw = pc_laplacian_stencil_weight(
                        -dx, -dy, -dz);
                    L_ij = raw / (2.0 * h_i);
                } else {
                    /* depth_delta == +1 (neighbour is finer) cannot be
                     * a coarser-depth constraint for the current depth,
                     * but we keep the general form for completeness.
                     */
                    const double h_j = cell_j.bounds.extent().x;
                    const double inv_hj = 1.0 / h_j;
                    const int dx = static_cast<int>(
                        std::round((center_j.x - center_i.x) * inv_hj));
                    const int dy = static_cast<int>(
                        std::round((center_j.y - center_i.y) * inv_hj));
                    const int dz = static_cast<int>(
                        std::round((center_j.z - center_i.z) * inv_hj));
                    if (dx < -4 || dx > 4 || dy < -4 || dy > 4 ||
                        dz < -4 || dz > 4) {
                        continue;
                    }
                    const double raw = pc_laplacian_stencil_weight(
                        dx, dy, dz);
                    L_ij = raw / (2.0 * h_j);
                }

                constraint += L_ij * x[j_global];
            }

            /* ---- Screening cross-depth couplings ----
             *
             * Screening is assembled as a sparse mass-like term and can
             * couple DOFs across depths (via overlapping B-spline
             * supports).  We treat already-solved coarser neighbours as
             * fixed and move them to the RHS.
             */
            const std::size_t s_start = screening.offsets[i_global];
            const std::size_t s_end = screening.offsets[i_global + 1];
            for (std::size_t k = s_start; k < s_end; ++k) {
                const std::int64_t j_dof = screening.neighbors[k];
                if (j_dof < 0) continue;
                const std::size_t j_global =
                    static_cast<std::size_t>(j_dof);

                const std::size_t cj = dof_to_cell[j_global];
                if (cells[cj].depth >= d) {
                    continue;
                }
                constraint += screening.weights[k] * x[j_global];
            }

            /* Step 2: Effective RHS for depth d. */
            rhs_eff[i_local] = b[i_global] - constraint;
        }

        /* ----------------------------------------------------------
         * Step 3: Solve the same-depth system A_dd x_d = rhs_eff.
         *
         * We use a local PCG solve with a Jacobi preconditioner.
         * The mat-vec is the restricted apply_operator_depth().
         *
         * This is the depth-local PCG inner loop of ToG13 Algorithm 1.
         * ---------------------------------------------------------- */
        std::vector<double> x_local(n_local, 0.0);
        for (std::size_t i_local = 0; i_local < n_local; ++i_local) {
            x_local[i_local] = x[dof_begin + i_local];
        }

        /* Build the Jacobi diagonal for A_dd.
         *
         * The Laplacian diagonal is L(0,0,0,h_i) and the screening
         * diagonal is stored in screening.diagonal[i].
         */
        std::vector<double> diag_local(n_local, 0.0);
        for (std::size_t i_global = dof_begin; i_global < dof_end;
             ++i_global) {
            const std::size_t i_local = i_global - dof_begin;
            const std::size_t ci = dof_to_cell[i_global];
            const double h = cells[ci].bounds.extent().x;
            diag_local[i_local] =
                laplacian_stencil_weight(0, 0, 0, h) +
                screening.diagonal[i_global];
        }

        /* Initial residual r = rhs_eff - A_dd * x_local. */
        std::vector<double> Ax_local;
        apply_operator_depth(x_local, cells, cell_to_dof, dof_to_cell,
                             stencil_offsets, stencil_neighbors,
                             stencil_depth_deltas, screening,
                             dof_begin, dof_end, Ax_local);

        std::vector<double> r_local(n_local, 0.0);
        for (std::size_t i = 0; i < n_local; ++i) {
            r_local[i] = rhs_eff[i] - Ax_local[i];
        }

        const double rhs_norm = vec_norm(rhs_eff, n_local);
        const double abs_tol =
            (rhs_norm > 1e-30) ? tol_d * rhs_norm : tol_d;

        double r_norm = vec_norm(r_local, n_local);
        if (r_norm >= abs_tol && max_iters_per_depth > 0) {
            /* Preconditioned residual z = M^{-1} r. */
            std::vector<double> z_local;
            apply_jacobi(r_local, diag_local, z_local, n_local);

            /* Search direction p = z. */
            std::vector<double> p_local = z_local;
            double rz = vec_dot(r_local, z_local, n_local);

            std::vector<double> q_local;

            bool depth_converged = false;
            for (std::size_t iter = 0; iter < max_iters_per_depth; ++iter) {
                /* q = A_dd * p. */
                apply_operator_depth(p_local, cells, cell_to_dof,
                                     dof_to_cell, stencil_offsets,
                                     stencil_neighbors,
                                     stencil_depth_deltas, screening,
                                     dof_begin, dof_end, q_local);

                /* alpha = (r·z) / (p·q). */
                const double pq = vec_dot(p_local, q_local, n_local);
                if (std::abs(pq) < 1e-30) {
                    /* Numerical breakdown (should be rare for SPD). */
                    break;
                }
                const double alpha = rz / pq;

                /* x += alpha * p. */
                vec_axpy(x_local, alpha, p_local, n_local);

                /* r -= alpha * q. */
                vec_axpy(r_local, -alpha, q_local, n_local);

                r_norm = vec_norm(r_local, n_local);
                result.iterations += 1;
                if (r_norm < abs_tol) {
                    depth_converged = true;
                    break;
                }

                /* z = M^{-1} r. */
                apply_jacobi(r_local, diag_local, z_local, n_local);

                /* beta = (r·z_new) / (r·z_old). */
                const double rz_new = vec_dot(r_local, z_local, n_local);
                const double beta =
                    (std::abs(rz) > 1e-30) ? rz_new / rz : 0.0;

                /* p = z + beta * p. */
                for (std::size_t i = 0; i < n_local; ++i) {
                    p_local[i] = z_local[i] + beta * p_local[i];
                }
                rz = rz_new;
            }

            if (!depth_converged) {
                /* We keep going even if one depth fails to converge,
                 * but we report the overall solve as non-converged.
                 * (ToG13 recommends cascadic iterations; Phase 21f
                 * implements a single pass depth-by-depth.) */
                result.converged = false;
            }
        }

        /* Write back the solved depth-d values into the global x. */
        for (std::size_t i_local = 0; i_local < n_local; ++i_local) {
            x[dof_begin + i_local] = x_local[i_local];
        }
    }

    /* --------------------------------------------------------------
     * Report a global residual norm for the final x.
     *
     * Even though we solved per-depth blocks, users care about the
     * residual of the full operator A.
     * -------------------------------------------------------------- */
    std::vector<double> Ax;
    apply_operator(x, cells, cell_to_dof, dof_to_cell, stencil_offsets,
                   stencil_neighbors, stencil_depth_deltas, screening,
                   n_dofs, Ax);

    std::vector<double> r;
    compute_residual(Ax, b, r, n_dofs);
    const double b_norm = vec_norm(b, n_dofs);
    const double r_global_norm = vec_norm(r, n_dofs);
    result.residual_norm = r_global_norm / (b_norm + 1e-30);

    return result;
}

/**
 * @brief High-level solve: builds operator and solves Ax = b.
 *
 * This is the main entry point for the Poisson solver.  It
 * takes the sample positions (for screening), cells, domain,
 * and the RHS vector b, and returns the solution x.
 *
 * @param positions      Sample positions (N points).
 * @param n_samples      Number of samples.
 * @param cells          Octree cells.
 * @param domain         Domain bounding box.
 * @param base_resolution Top-level cells per axis.
 * @param max_depth      Maximum octree depth.
 * @param alpha          Screening weight.
 * @param b              Right-hand side (size n_dofs).
 * @param max_iters      Maximum CG iterations.
 * @param tol            Relative residual tolerance.
 * @param x              Output solution (size n_dofs).
 * @return Solver result.
 */
inline SolverResult solve_poisson(
    const Vector3d *positions,
    std::size_t n_samples,
    const std::vector<OctreeCell> &cells,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t max_depth,
    double alpha,
    const std::vector<double> &b,
    std::size_t max_iters,
    double tol,
    std::vector<double> &x) {

    /* Build DOF indexing. */
    std::vector<std::int64_t> cell_to_dof;
    std::vector<std::size_t> dof_to_cell;
    assign_dof_indices(cells, cell_to_dof, dof_to_cell);
    const std::size_t n_dofs = dof_to_cell.size();

    /* Build spatial hash. */
    PoissonLeafHash hash;
    hash.build(cells, domain, max_depth, base_resolution);

    /* Build stencils. */
    std::vector<std::size_t> stencil_offsets;
    std::vector<std::int64_t> stencil_neighbors;
    std::vector<int> stencil_depth_deltas;
    enumerate_stencils(cells, cell_to_dof, dof_to_cell,
                       domain, base_resolution, max_depth,
                       stencil_offsets, stencil_neighbors,
                       &stencil_depth_deltas);

    /* Accumulate screening. */
    ScreeningData screening;
    accumulate_screening(positions, n_samples, alpha,
                         hash, cells, cell_to_dof, dof_to_cell,
                         n_dofs, base_resolution, screening);

    /* Initialise x to zero. */
    x.assign(n_dofs, 0.0);

    /* Solve with PCG. */
    return solve_pcg(b, cells, cell_to_dof, dof_to_cell,
                     stencil_offsets, stencil_neighbors,
                     stencil_depth_deltas,
                     screening, n_dofs, max_iters, tol, x);
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_POISSON_SOLVER_HPP_
