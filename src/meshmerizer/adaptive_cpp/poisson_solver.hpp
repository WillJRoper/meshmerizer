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
            const Vector3d center_j = cells[cj].bounds.center();
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

            if (dx < -1 || dx > 1 ||
                dy < -1 || dy > 1 ||
                dz < -1 || dz > 1) {
                continue;
            }

            off_diag_sum +=
                laplacian_stencil_weight(dx, dy, dz, h_i)
                * x[j];
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
    enumerate_stencils(cells, cell_to_dof, dof_to_cell,
                       domain, base_resolution, max_depth,
                       stencil_offsets, stencil_neighbors);

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
                     screening, n_dofs, max_iters, tol, x);
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_POISSON_SOLVER_HPP_
