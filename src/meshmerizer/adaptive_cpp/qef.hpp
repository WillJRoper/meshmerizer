/**
 * @file qef.hpp
 * @brief QEF (Quadratic Error Function) assembly and leaf vertex solve for
 * dual contouring.
 *
 * The QEF for a leaf cell accumulates one tangent-plane constraint per
 * Hermite sample:
 *
 *   E(x) = sum_i ( n_i . (x - p_i) )^2
 *
 * where p_i is a surface crossing position and n_i is the outward unit normal
 * at that crossing. Minimizing E places the representative vertex at the point
 * that best satisfies all the tangent-plane constraints simultaneously.
 *
 * Minimizing E is equivalent to solving the normal equations of the
 * least-squares system  A x = b  where each row of A is one n_i and the
 * corresponding entry of b is  d_i = n_i . p_i:
 *
 *   (A^T A) x = A^T b
 *   (sum_i n_i n_i^T) x = sum_i d_i n_i
 *
 * The 3x3 matrix  A^T A  is symmetric and positive semi-definite.
 * ``QEFAccumulator`` builds it incrementally from samples.  ``solve_qef_for_leaf``
 * solves the resulting system, falls back to the sample centroid when the
 * matrix is rank-deficient, and clamps the result to the cell bounding box.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_QEF_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_QEF_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "bounding_box.hpp"
#include "hermite.hpp"
#include "mesh.hpp"
#include "vector3d.hpp"

/**
 * @brief Accumulates the QEF normal equations from a set of Hermite samples.
 *
 * Each call to ``add_sample`` updates the symmetric 3x3 matrix and 3-vector
 * that represent  A^T A  and  A^T b  respectively, and also tracks the sample
 * centroid and mean normal for use in fallback and normal estimation.
 *
 * Samples with a zero-length normal (degenerate crossings from Phase 7) are
 * silently ignored and do not contribute to the accumulated system.
 */
struct QEFAccumulator {
    /// Symmetric 3x3 matrix A^T A = sum_i n_i n_i^T.
    std::array<std::array<double, 3>, 3> mat;
    /// Right-hand side A^T b = sum_i (n_i . p_i) n_i.
    std::array<double, 3> rhs;
    /// Running sum of valid sample positions (divided by count at solve time).
    Vector3d position_sum;
    /// Running sum of valid sample normals (normalized at solve time).
    Vector3d normal_sum;
    /// Number of valid (non-zero-normal) samples accumulated.
    std::uint32_t sample_count;

    /**
     * @brief Construct a zero-initialized accumulator.
     */
    QEFAccumulator()
        : mat{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}},
          rhs{0.0, 0.0, 0.0},
          position_sum{0.0, 0.0, 0.0},
          normal_sum{0.0, 0.0, 0.0},
          sample_count(0U) {}

    /**
     * @brief Add one Hermite sample to the accumulated QEF system.
     *
     * Samples whose normal has zero length are skipped because they carry no
     * tangent-plane information.
     *
     * @param sample Hermite sample to add.
     */
    void add_sample(const HermiteSample &sample) {
        const double nx = sample.normal.x;
        const double ny = sample.normal.y;
        const double nz = sample.normal.z;

        // Zero-normal samples are degenerate; skip them.
        if (nx == 0.0 && ny == 0.0 && nz == 0.0) {
            return;
        }

        // d_i = n_i . p_i (distance term for this tangent plane).
        const double d =
            nx * sample.position.x + ny * sample.position.y +
            nz * sample.position.z;

        // Accumulate the outer product n_i n_i^T into the matrix.
        mat[0][0] += nx * nx;
        mat[0][1] += nx * ny;
        mat[0][2] += nx * nz;
        mat[1][0] += ny * nx;
        mat[1][1] += ny * ny;
        mat[1][2] += ny * nz;
        mat[2][0] += nz * nx;
        mat[2][1] += nz * ny;
        mat[2][2] += nz * nz;

        // Accumulate d_i * n_i into the right-hand side.
        rhs[0] += d * nx;
        rhs[1] += d * ny;
        rhs[2] += d * nz;

        // Accumulate position and normal sums for fallback and output normal.
        position_sum.x += sample.position.x;
        position_sum.y += sample.position.y;
        position_sum.z += sample.position.z;
        normal_sum.x += nx;
        normal_sum.y += ny;
        normal_sum.z += nz;

        ++sample_count;
    }

    /**
     * @brief Return the centroid of all valid sample positions.
     *
     * Returns the zero vector when no valid samples have been accumulated.
     */
    Vector3d mass_point() const {
        if (sample_count == 0U) {
            return {0.0, 0.0, 0.0};
        }
        const double inv = 1.0 / static_cast<double>(sample_count);
        return {
            position_sum.x * inv,
            position_sum.y * inv,
            position_sum.z * inv,
        };
    }

    /**
     * @brief Return the normalized mean of all valid sample normals.
     *
     * Returns the zero vector when no valid samples have been accumulated or
     * the mean normal has zero length.
     */
    Vector3d mean_normal() const {
        const double magnitude =
            std::sqrt(normal_sum.x * normal_sum.x +
                      normal_sum.y * normal_sum.y +
                      normal_sum.z * normal_sum.z);
        if (magnitude < 1e-12) {
            return {0.0, 0.0, 0.0};
        }
        return {
            normal_sum.x / magnitude,
            normal_sum.y / magnitude,
            normal_sum.z / magnitude,
        };
    }
};

/**
 * @brief Solve a 3x3 linear system using Gaussian elimination with
 * partial pivoting.
 *
 * ``mat`` and ``rhs`` are taken by value (copied), so the caller's
 * originals are not modified.  If any pivot falls below
 * ``pivot_threshold`` the system is considered rank-deficient and the
 * function returns ``false`` without writing a solution.
 *
 * @param mat 3x3 coefficient matrix (copied internally).
 * @param rhs 3-element right-hand side (copied internally).
 * @param solution Output 3-element solution vector.
 * @param pivot_threshold Minimum acceptable pivot magnitude.
 * @return ``true`` when a unique solution was found.
 */
inline bool solve_3x3_system(
    std::array<std::array<double, 3>, 3> mat,
    std::array<double, 3> rhs,
    std::array<double, 3> &solution,
    double pivot_threshold = 1e-10) {
    // Build an augmented 3x4 matrix [mat | rhs] and apply row operations.
    std::array<std::array<double, 4>, 3> aug;
    for (int row = 0; row < 3; ++row) {
        aug[row][0] = mat[row][0];
        aug[row][1] = mat[row][1];
        aug[row][2] = mat[row][2];
        aug[row][3] = rhs[row];
    }

    // Forward elimination with partial pivoting.
    for (int col = 0; col < 3; ++col) {
        // Find the row with the largest absolute value in this column.
        int pivot_row = col;
        double max_abs = std::abs(aug[col][col]);
        for (int row = col + 1; row < 3; ++row) {
            const double abs_val = std::abs(aug[row][col]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
                pivot_row = row;
            }
        }

        if (max_abs < pivot_threshold) {
            // The system is singular or nearly so: the QEF constraints do not
            // uniquely determine a position in this column direction.
            return false;
        }

        // Swap the pivot row into position.
        if (pivot_row != col) {
            std::swap(aug[col], aug[pivot_row]);
        }

        // Eliminate entries below the pivot.
        for (int row = col + 1; row < 3; ++row) {
            const double factor = aug[row][col] / aug[col][col];
            for (int k = col; k < 4; ++k) {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    // Back substitution.
    for (int row = 2; row >= 0; --row) {
        double sum = aug[row][3];
        for (int col = row + 1; col < 3; ++col) {
            sum -= aug[row][col] * solution[col];
        }
        solution[row] = sum / aug[row][row];
    }

    return true;
}

/**
 * @brief Solve the QEF and produce a representative MeshVertex for one leaf.
 *
 * The algorithm:
 *
 * 1. Accumulate the QEF normal equations from all provided Hermite samples.
 * 2. Apply Tikhonov regularization: add ``lambda * I`` to the matrix and
 *    ``lambda * cell_center`` to the right-hand side.  This biases the
 *    solution toward the cell center when tangent-plane constraints are
 *    ill-conditioned (e.g., nearly parallel normals), preventing the
 *    vertex from wandering to extreme positions.  The technique follows
 *    Schaefer et al. 2007 ("Dual Contouring: The Secret Sauce").
 * 3. Attempt to solve the regularized 3x3 system with Gaussian elimination.
 * 4. If the system is still rank-deficient after regularization (should be
 *    rare), fall back to the sample centroid (mass point).
 * 5. Clamp the result to the cell bounding box so the vertex stays inside
 *    the cell it belongs to.
 * 6. Assign the normalized mean sample normal as the vertex normal.
 *
 * A leaf with no Hermite samples (or only degenerate zero-normal samples)
 * receives the cell center as its vertex position and a zero normal.
 *
 * @param samples Hermite samples for this leaf (from Phase 7).
 * @param bounds Cell bounding box used for clamping and regularization.
 * @param regularization_weight Tikhonov regularization strength.  Larger
 *     values pull the vertex more strongly toward the cell center.  A
 *     value of 0.0 disables regularization (original behavior).
 *     Default: 0.1, which provides gentle bias without distorting
 *     well-constrained vertices.
 * @return Representative mesh vertex for this leaf.
 */
inline MeshVertex solve_qef_for_leaf(const std::vector<HermiteSample> &samples,
                                     const BoundingBox &bounds,
                                     double regularization_weight = 0.1) {
    // Assemble the normal equations.
    QEFAccumulator accumulator;
    for (const HermiteSample &sample : samples) {
        accumulator.add_sample(sample);
    }

    // Degenerate leaf: no usable samples at all.
    if (accumulator.sample_count == 0U) {
        return {bounds.center(), {0.0, 0.0, 0.0}};
    }

    // Apply Tikhonov regularization: add lambda * I to A^T A and
    // lambda * cell_center to A^T b.  This makes the system
    //
    //   (A^T A + lambda I) x = A^T b + lambda c
    //
    // which biases the solution toward the cell center c when the
    // tangent-plane constraints alone are insufficient to uniquely
    // determine a position.  Well-constrained vertices (where A^T A
    // already has large eigenvalues) are barely affected.
    if (regularization_weight > 0.0) {
        const Vector3d center = bounds.center();
        accumulator.mat[0][0] += regularization_weight;
        accumulator.mat[1][1] += regularization_weight;
        accumulator.mat[2][2] += regularization_weight;
        accumulator.rhs[0] += regularization_weight * center.x;
        accumulator.rhs[1] += regularization_weight * center.y;
        accumulator.rhs[2] += regularization_weight * center.z;
    }

    // Attempt the 3x3 solve.
    std::array<double, 3> solution{0.0, 0.0, 0.0};
    const bool solved = solve_3x3_system(
        accumulator.mat, accumulator.rhs, solution);

    // Use the solved position, or the mass point if the system was singular.
    Vector3d position;
    if (solved) {
        position = {solution[0], solution[1], solution[2]};
    } else {
        position = accumulator.mass_point();
    }

    // Clamp the vertex to the cell bounding box. The QEF minimizer can
    // wander outside the cell when constraints are clustered near one face.
    position.x =
        position.x < bounds.min.x ? bounds.min.x
        : position.x > bounds.max.x ? bounds.max.x
        : position.x;
    position.y =
        position.y < bounds.min.y ? bounds.min.y
        : position.y > bounds.max.y ? bounds.max.y
        : position.y;
    position.z =
        position.z < bounds.min.z ? bounds.min.z
        : position.z > bounds.max.z ? bounds.max.z
        : position.z;

    return {position, accumulator.mean_normal()};
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_QEF_HPP_
