/**
 * @file poisson_rhs.hpp
 * @brief Vector field splatting and RHS assembly for screened Poisson
 *        surface reconstruction using degree-2 (quadratic) B-splines.
 *
 * @par References
 * - Kazhdan, M., Bolitho, M. & Hoppe, H. "Poisson Surface
 *   Reconstruction", *Proc. SGP* (2006), Section 3. The smoothed
 *   normal field V approximates the gradient of the smoothed
 *   indicator function: \f$\nabla\chi\approx\vec{V}\f$, yielding the
 *   Poisson equation \f$\Delta\chi = \nabla\cdot\vec{V}\f$. The weak
 *   form RHS is \f$b_i=\langle\vec{V},\nabla B_i\rangle\f$.
 * - Kazhdan, M. & Hoppe, H. "Screened Poisson Surface
 *   Reconstruction", *ACM Trans. Graph.* 32(3), Art. 29 (2013),
 *   Section 3. The vector field V is built by splatting oriented
 *   normals into the B-spline basis.
 * - Unser, M. "Splines: A Perfect Fit for Signal and Image
 *   Processing", *IEEE Signal Processing Magazine* 16(6), 22–38
 *   (1999). (B-spline support properties and basic identities.)
 * - de Boor, C. *A Practical Guide to Splines*, Revised Edition,
 *   Springer (2001). (B-spline definitions and convolution support.)
 * - Reference implementation: https://github.com/mkazhdan/PoissonRecon
 *   (MIT licence).
 *
 * @par Algorithm
 * 1. **Splat normals** — For each oriented sample \f$(p_s,n_s)\f$:
 *    - Enumerate the (up to 27) overlapping DOFs whose quadratic
 *      B-spline support covers \f$p_s\f$.
 *    - For each overlapping DOF \f$j\f$, accumulate:
 *      \f$\vec{V}_j \mathrel{+}= B_j(p_s)\,n_s\,w_s\f$.
 *
 * 2. **Assemble RHS (Galerkin)** — With
 *    \f$\vec{V}(x)=\sum_j \vec{V}_j B_j(x)\f$, the weak RHS becomes:
 *    \f$b_i=\int \vec{V}(x)\cdot\nabla B_i(x)\,dx
 *          =\sum_{j\in\mathrm{stencil}(i)} \vec{V}_j\cdot\vec{G}_{ij}\f$,
 *    where \f$\vec{G}_{ij}=\int B_j(x)\,\nabla B_i(x)\,dx\f$ depends
 *    only on the relative offset \f$(i-j)\f$ and the cell width \f$h\f$.
 *
 * For degree-2 tensor-product B-splines on a uniform grid (unit knot
 * spacing in reference coordinates), the 1-D integrals factor and the
 * 3-D gradient inner product is:
 * \f[
 *   G_x = S(dx)\,M(dy)\,M(dz)\,h^2,\quad
 *   G_y = M(dx)\,S(dy)\,M(dz)\,h^2,\quad
 *   G_z = M(dx)\,M(dy)\,S(dz)\,h^2,
 * \f]
 * where \f$M(d)=\int B_2(t)B_2(t-d)dt\f$ and
 * \f$S(d)=\int B_2'(t)B_2(t-d)dt\f$. The \f$h^2\f$ arises from the
 * volume element \f$h^3\f$ and the chain rule factor \f$1/h\f$ in the
 * gradient.
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
 * Section 1: Precomputed 1-D integrals for degree-2 B-splines
 * =================================================================== */

/**
 * @brief 1-D mass (overlap) integral \f$M(d)\f$ for quadratic
 *        (degree-2) B-splines with unit knot spacing.
 *
 * \f$M(d)=\int_{-\infty}^{\infty} B_2(t)\,B_2(t-d)\,dt\f$.
 *
 * Nonzero only for \f$d\in\{-2,-1,0,1,2\}\f$:
 * - \f$M(0)=11/20\f$
 * - \f$M(\pm1)=13/60\f$
 * - \f$M(\pm2)=1/120\f$
 *
 * These constants are the autocorrelation of the quadratic B-spline
 * (cf. Unser 1999; de Boor 2001) evaluated at integer shifts.
 *
 * @param d Relative offset in cell-width units (integer).
 * @return Integral value.
 */
inline double mass_integral_1d(int d) {
    switch (d) {
        case 0:
            return 11.0 / 20.0;
        case 1:
        case -1:
            return 13.0 / 60.0;
        case 2:
        case -2:
            return 1.0 / 120.0;
        default:
            return 0.0;
    }
}

/**
 * @brief 1-D stiffness (gradient-gradient) integral \f$K(d)\f$ for
 *        quadratic (degree-2) B-splines with unit knot spacing.
 *
 * \f$K(d)=\int_{-\infty}^{\infty} B_2'(t)\,B_2'(t-d)\,dt\f$.
 *
 * Nonzero only for \f$d\in\{-2,-1,0,1,2\}\f$:
 * - \f$K(0)=1\f$
 * - \f$K(\pm1)=-1/3\f$
 * - \f$K(\pm2)=-1/6\f$
 *
 * The sum over shifts vanishes
 * \f$\sum_{d=-2}^{2} K(d)=0\f$, consistent with constants being in
 * the nullspace of the gradient operator.
 *
 * @param d Relative offset in cell-width units.
 * @return Integral value.
 */
inline double stiffness_integral_1d(int d) {
    switch (d) {
        case 0:
            return 1.0;
        case 1:
        case -1:
            return -1.0 / 3.0;
        case 2:
        case -2:
            return -1.0 / 6.0;
        default:
            return 0.0;
    }
}

/**
 * @brief 1-D gradient–value integral \f$S(d)\f$ for quadratic
 *        (degree-2) B-splines with unit knot spacing.
 *
 * \f$S(d)=\int_{-\infty}^{\infty} B_2'(t)\,B_2(t-d)\,dt\f$.
 *
 * Nonzero only for \f$d\in\{-2,-1,0,1,2\}\f$:
 * - \f$S(0)=0\f$
 * - \f$S(+1)=-5/12\f$, \f$S(-1)=+5/12\f$
 * - \f$S(+2)=-1/24\f$, \f$S(-2)=+1/24\f$
 *
 * Note: \f$S(d)=-S(-d)\f$ (antisymmetry of the derivative).
 *
 * @param d Relative offset in cell-width units.
 * @return Integral value.
 */
inline double grad_value_integral_1d(int d) {
    switch (d) {
        case 1:
            return -5.0 / 12.0;
        case -1:
            return 5.0 / 12.0;
        case 2:
            return -1.0 / 24.0;
        case -2:
            return 1.0 / 24.0;
        default:
            return 0.0;
    }
}

/* ===================================================================
 * Section 1b: Parent-child cross-depth 1-D integrals (degree 2)
 * ===================================================================
 *
 * These tables store the 1-D B-spline overlap integrals between a
 * parent DOF at depth d-1 (cell width 2h) and a child DOF at depth d
 * (cell width h).  The offset j is the distance from the parent centre
 * to the child centre in units of h (the finer cell width).
 *
 * Raw integrals (before physical scaling):
 *   M_pc(j) = int B2(u/2) * B2(u - j) du    (mass)
 *   K_pc(j) = int B2'(u/2) * B2'(u - j) du  (stiffness)
 *   S_pc(j) = int B2'(u/2) * B2(u - j) du   (gradient-value)
 *
 * Physical scaling to obtain Galerkin integrals in world coordinates:
 *   Mass:        h * M_pc(j)       (volume element)
 *   Stiffness:   (1 / (2h)) * K_pc(j)  (chain rules: 1/(2h) * 1/h * h)
 *   Grad-Value:  (1 / 2) * S_pc(j)     (chain rule on parent: 1/(2h) * h)
 *
 * Nonzero for j in {-4, ..., +4} (9 entries).  The j = +/-4 entries
 * are tiny (~1e-4) and can optionally be dropped, but we keep them for
 * correctness.  In 3-D the full stencil is 9^3 = 729 entries.
 *
 * Verification checksums (cf. adaptive-plan.md Phase 21):
 *   sum_j M_pc(j) = 2.0   (= integral of B2(u/2) du)
 *   sum_j K_pc(j) = 0.0   (derivative identity)
 *   S_pc(-j) = -S_pc(j)   (antisymmetry)
 *
 * References:
 * - Kazhdan & Hoppe (ToG 2013), Section 4: multi-depth FEM stencils
 *   with overlapping B-spline supports across resolution levels.
 * - PoissonRecon source: BSplineData.inl, SystemCoefficients<2>.
 */

/**
 * @brief Parent-child 1-D mass integral M_pc(j) for degree-2
 *        B-splines with width ratio 2:1.
 *
 * M_pc(j) = int B2(u/2) * B2(u - j) du, where parent has width 2
 * and child has width 1 in reference coordinates.
 *
 * @param j  Offset from parent centre to child centre in child-width
 *           units (integer, -4..+4).
 * @return Raw integral value (multiply by h for physical scaling).
 */
inline double pc_mass_integral_1d(int j) {
    switch (j) {
        case 0:
            return 1761.0 / 2560.0;
        case 1:
        case -1:
            return 31.0 / 64.0;
        case 2:
        case -2:
            return 599.0 / 3840.0;
        case 3:
        case -3:
            return 1.0 / 64.0;
        case 4:
        case -4:
            return 1.0 / 15360.0;
        default:
            return 0.0;
    }
}

/**
 * @brief Parent-child 1-D stiffness integral K_pc(j) for degree-2
 *        B-splines with width ratio 2:1.
 *
 * K_pc(j) = int B2'(u/2) * B2'(u - j) du.
 *
 * @param j  Offset in child-width units (-4..+4).
 * @return Raw integral value (multiply by 1/(2h) for physical
 *         scaling).
 */
inline double pc_stiffness_integral_1d(int j) {
    switch (j) {
        case 0:
            return 15.0 / 16.0;
        case 1:
        case -1:
            return 1.0 / 4.0;
        case 2:
        case -2:
            return -11.0 / 24.0;
        case 3:
        case -3:
            return -1.0 / 4.0;
        case 4:
        case -4:
            return -1.0 / 96.0;
        default:
            return 0.0;
    }
}

/**
 * @brief Parent-child 1-D gradient-value integral S_pc(j) for
 *        degree-2 B-splines with width ratio 2:1.
 *
 * S_pc(j) = int B2'(u/2) * B2(u - j) du.
 * Antisymmetric: S_pc(-j) = -S_pc(j).
 *
 * @param j  Offset in child-width units (-4..+4).
 * @return Raw integral value (multiply by 1/2 for physical scaling).
 */
inline double pc_grad_value_integral_1d(int j) {
    switch (j) {
        case 0:
            return 0.0;
        case 1:
            return -89.0 / 128.0;
        case -1:
            return 89.0 / 128.0;
        case 2:
            return -191.0 / 384.0;
        case -2:
            return 191.0 / 384.0;
        case 3:
            return -13.0 / 128.0;
        case -3:
            return 13.0 / 128.0;
        case 4:
            return -1.0 / 768.0;
        case -4:
            return 1.0 / 768.0;
        default:
            return 0.0;
    }
}

/**
 * @brief Parent-child 3-D Laplacian (stiffness) stencil weight.
 *
 * For a parent DOF at depth d-1 and a child DOF at depth d with
 * 3-D offset (dx, dy, dz) in child-width units:
 *
 *   L_pc = K_pc(dx)*M_pc(dy)*M_pc(dz)
 *        + M_pc(dx)*K_pc(dy)*M_pc(dz)
 *        + M_pc(dx)*M_pc(dy)*K_pc(dz)
 *
 * Physical scaling: multiply by 1/(2h) where h is the child cell
 * width.
 *
 * @param dx  Offset in x (child-width units, -4..+4).
 * @param dy  Offset in y.
 * @param dz  Offset in z.
 * @return Raw 3-D stiffness weight.
 */
inline double pc_laplacian_stencil_weight(int dx, int dy, int dz) {
    const double mx = pc_mass_integral_1d(dx);
    const double my = pc_mass_integral_1d(dy);
    const double mz = pc_mass_integral_1d(dz);
    const double kx = pc_stiffness_integral_1d(dx);
    const double ky = pc_stiffness_integral_1d(dy);
    const double kz = pc_stiffness_integral_1d(dz);
    return kx * my * mz + mx * ky * mz + mx * my * kz;
}

/**
 * @brief Parent-child 3-D mass stencil weight (for screening).
 *
 * M_pc_3D = M_pc(dx) * M_pc(dy) * M_pc(dz).
 *
 * Physical scaling: multiply by h (child cell width).
 *
 * @param dx  Offset in x (child-width units, -4..+4).
 * @param dy  Offset in y.
 * @param dz  Offset in z.
 * @return Raw 3-D mass weight.
 */
inline double pc_mass_stencil_weight(int dx, int dy, int dz) {
    return pc_mass_integral_1d(dx) * pc_mass_integral_1d(dy) *
           pc_mass_integral_1d(dz);
}

/**
 * @brief Parent-child 3-D gradient inner product vector.
 *
 * For RHS assembly with cross-depth DOF pairs:
 *   G_pc.x = S_pc(dx) * M_pc(dy) * M_pc(dz)
 *   G_pc.y = M_pc(dx) * S_pc(dy) * M_pc(dz)
 *   G_pc.z = M_pc(dx) * M_pc(dy) * S_pc(dz)
 *
 * Physical scaling: multiply by h*h/2 where h is the child cell
 * width.  (Volume element h^3, parent gradient 1/(2h), child
 * value 1: net = h^3 / (2h) = h^2 / 2.)
 *
 * @param dx  Offset in x (child-width units, -4..+4).
 * @param dy  Offset in y.
 * @param dz  Offset in z.
 * @return Gradient inner product vector (unscaled).
 */
inline Vector3d pc_gradient_inner_product(int dx, int dy, int dz) {
    const double mx = pc_mass_integral_1d(dx);
    const double my = pc_mass_integral_1d(dy);
    const double mz = pc_mass_integral_1d(dz);
    const double sx = pc_grad_value_integral_1d(dx);
    const double sy = pc_grad_value_integral_1d(dy);
    const double sz = pc_grad_value_integral_1d(dz);
    return {
        sx * my * mz,
        mx * sy * mz,
        mx * my * sz,
    };
}

/* ===================================================================
 * Section 2: 3-D gradient inner product G_{ij} (same-depth)
 * =================================================================== */

/**
 * @brief Compute the 3-D gradient inner product vector G for a
 *        given relative cell offset (dx, dy, dz).
 *
 * G is a 3-component vector where:
 *   G.x = S(dx) * M(dy) * M(dz) * h^2
 *   G.y = M(dx) * S(dy) * M(dz) * h^2
 *   G.z = M(dx) * M(dy) * S(dz) * h^2
 *
 * This is the integral of B_j(x) * grad B_i(x) over the domain,
 * where cell j is offset from cell i by (dx, dy, dz) cell widths.
 *
 * For quadratic B-splines, overlaps are nonzero for offsets up to
 * \f$\pm 2\f$ in each axis, corresponding to a \f$5^3=125\f$ stencil
 * in 3-D (including edges/corners).
 *
 * The factor of \f$h^3\f$ from the volume element and the factor of
 * \f$1/h\f$ from the gradient chain rule combine to give \f$h^2\f$
 * overall.
 *
 * @param dx  Offset in x (integer: -2..+2).
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
     * introduces a 1/h factor (chain rule on the B-spline argument
     * x/h). Net scaling is h^2.
     *
     * We express this as h*h to keep the arithmetic transparent.
     * (The remaining dimensionless factors come from the tabulated
     * 1-D integrals on the unit grid.) */
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
 *        point and return their indices with quadratic weights.
 *
 * For a degree-2 B-spline, the 1-D support in reference coordinates
 * is \f$[-3/2,\,3/2]\f$ (Unser 1999). In fine-grid index units, a cell
 * with min-corner index \f$m\f$ has centre at \f$m+1/2\f$ and support
 * over \f$[m-1,\,m+2]\f$. Therefore, a point can overlap at most
 * \f$3^3=27\f$ DOFs.
 *
 * **Approach**: Rather than finding a "containing cell" first
 * (which fails when the hash quantization maps the point to a
 * grid coordinate that doesn't match any cell's min-corner key),
 * we directly compute which cell-aligned min-corner positions
 * could overlap the point. For each axis, we convert the point to
 * fine-grid coordinates, then probe the three candidate min-corner
 * indices \f$\{\lfloor t-1\rfloor,\lfloor t-1\rfloor+1,
 * \lfloor t-1\rfloor+2\}\f$. All \f$3^3=27\f$ combinations are
 * checked via the hash.
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

    /* For each axis, enumerate the three candidate cell min-corner
     * indices whose quadratic B-spline supports can overlap the
     * point.
     *
     * A cell with min-corner index m has centre at (m + 0.5) in
     * fine-grid units. The quadratic B-spline support is
     * |t - (m + 0.5)| < 3/2, i.e. t - 2 < m < t + 1.
     *
     * For non-integer t, the three integer solutions are:
     *   m = floor(t - 1), floor(t - 1) + 1, floor(t - 1) + 2.
     * This yields 3^3 = 27 candidates in 3-D. */
    const auto base_idx = [](double t) -> std::int64_t {
        return static_cast<std::int64_t>(std::floor(t - 1.0));
    };

    const std::int64_t ix0 = base_idx(tx);
    const std::int64_t iy0 = base_idx(ty);
    const std::int64_t iz0 = base_idx(tz);

    /* Probe all 27 combinations of {ix0, ix0+1, ix0+2} x ... These
     * are the candidate cell min-corner positions in fine-grid
     * coordinates. */
    for (int ox = 0; ox <= 2; ++ox) {
        for (int oy = 0; oy <= 2; ++oy) {
            for (int oz = 0; oz <= 2; ++oz) {
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

                /* Compute the quadratic B-spline weight for this
                 * candidate DOF. We evaluate in world space using
                 * the cell center and width, matching the basis
                 * used elsewhere in the Poisson system assembly. */
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

    /* Area weight per sample.  Each sample represents an oriented
     * surface element whose contribution to the B-spline divergence
     * must be scaled to produce an O(1) indicator function.
     *
     * The key insight (SGP06 Sec 3.1, PoissonRecon implementation):
     * the splatted vector field V_j should approximate the gradient
     * of the indicator function, which is a surface delta function
     * with magnitude O(1/h) where h is the cell width.  With N
     * samples on the surface and ~N*h^2 ≈ Area samples per cell
     * face, each sample contributes ~1 to V_j.  To get V_j ~ 1/h,
     * we scale each sample by 1/h = 2^depth / domain_extent.
     *
     * Since all DOFs are at max_depth, h is the fine cell width.
     * We approximate 1/h by fine_cells_per_axis / domain_extent,
     * but since the hash uses normalised coords, we can simply
     * use hash.inv_cell_size_x (which equals fine_per_axis /
     * domain_extent_x).  For cubic domains this is 1/h. */
    (void)n_samples;  /* Not used for weighting. */

    /* Compute 1/h for the fine-grid cell width.  This scales
     * the normal splatting so that V_j ~ O(1/h), producing an
     * O(1) indicator function after solving the Poisson system. */
    /* inv_cell_size_x = fine_per_axis / domain_extent_x = 1/h for
     * cubic domains at the leaf resolution. */
    const double inv_h = hash.inv_cell_size_x;

    ProgressBar progress("Splatting normals", n_samples);

    /* Per-sample temporary vectors to avoid repeated allocation.
     *
     * Note: each sample touches at most 27 DOFs for degree-2
     * B-splines. This implementation is serial for simplicity;
     * if profiling warrants, a parallel version should use atomics
     * or thread-local buffers with reduction. */
    std::vector<std::int64_t> dof_indices;
    std::vector<double> bweights;
    dof_indices.reserve(27);
    bweights.reserve(27);

    for (std::size_t s = 0; s < n_samples; ++s) {
        find_overlapping_dofs(positions[s], hash, cells,
                              cell_to_dof, dof_indices,
                              bweights, base_resolution);

        for (std::size_t k = 0; k < dof_indices.size(); ++k) {
            const std::size_t dof =
                static_cast<std::size_t>(dof_indices[k]);
            const double w = bweights[k] * inv_h;
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
 * @brief Compute the RHS vector \f$b\f$ using the Galerkin gradient
 *        inner product with degree-2 B-splines.
 *
 * The weak-form right-hand side (SGP06 Sec. 3) is
 * \f$b_i=\langle\vec{V},\nabla B_i\rangle
 *      =\int \vec{V}(x)\cdot\nabla B_i(x)\,dx\f$.
 *
 * With \f$\vec{V}(x)=\sum_j \vec{V}_j B_j(x)\f$, we obtain
 * \f$b_i=\sum_j \vec{V}_j\cdot\vec{G}_{ij}\f$, where
 * \f$\vec{G}_{ij}=\int B_j(x)\,\nabla B_i(x)\,dx\f$.
 *
 * For quadratic B-splines, \f$\vec{G}_{ij}\f$ is nonzero only when
 * the relative offset components satisfy \f$dx,dy,dz\in[-2,2]\f$,
 * i.e. a \f$5^3=125\f$ neighborhood in the uniform-grid case.
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
    /* cell_to_dof is part of the shared assembly interface.
     * It is not needed for RHS assembly when the stencil provides
     * neighbor DOF indices directly. */
    (void)cell_to_dof;

    rhs.assign(n_dofs, 0.0);

    ProgressBar progress("Assembling RHS", n_dofs);

    for (std::size_t i = 0; i < n_dofs; ++i) {
        const std::size_t ci = dof_to_cell[i];
        const OctreeCell &cell_i = cells[ci];
        const Vector3d center_i = cell_i.bounds.center();
        const double h_i = cell_i.bounds.extent().x;

        /* Galerkin RHS: accumulate contributions from the neighbor
         * DOFs in the provided stencil.
         *
         * We compute integer offsets by comparing cell centers in
         * units of the local cell width h_i. For the uniform-leaf
         * case, offsets are exact integers; for adaptive boundaries,
         * rounding combined with the [-2,2] guard prevents spurious
         * far-field coupling. */
        double accum = 0.0;
        const double inv_h = 1.0 / h_i;

        const std::size_t start = stencil_offsets[i];
        const std::size_t end = stencil_offsets[i + 1];

        for (std::size_t k = start; k < end; ++k) {
            const std::int64_t j_dof = stencil_neighbors[k];
            const std::size_t cj = dof_to_cell[
                static_cast<std::size_t>(j_dof)];
            const OctreeCell &cell_j = cells[cj];
            const Vector3d center_j = cell_j.bounds.center();

            const int dx = static_cast<int>(
                std::round((center_j.x - center_i.x) * inv_h));
            const int dy = static_cast<int>(
                std::round((center_j.y - center_i.y) * inv_h));
            const int dz = static_cast<int>(
                std::round((center_j.z - center_i.z) * inv_h));

            /* Quadratic B-spline overlaps extend out to ±2 cells
             * in each axis. Neighbors outside this range have zero
             * contribution by construction of the 1-D tables. */
            if (dx < -2 || dx > 2 ||
                dy < -2 || dy > 2 ||
                dz < -2 || dz > 2) {
                continue;
            }

            const Vector3d &Vj = v_field[
                static_cast<std::size_t>(j_dof)];

            /* \f$\vec{G}_{ij}\f$ is a vector-valued inner product.
             * Accumulate \f$\vec{V}_j\cdot\vec{G}_{ij}\f$.
             *
             * The returned G already includes the \f$h^2\f$ scaling
             * from the Jacobian and chain rule. */
            const Vector3d G =
                gradient_inner_product(dx, dy, dz, h_i);
            accum += Vj.x * G.x + Vj.y * G.y + Vj.z * G.z;
        }

        rhs[i] = accum;
        progress.tick();
    }

    progress.finish();
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_POISSON_RHS_HPP_
