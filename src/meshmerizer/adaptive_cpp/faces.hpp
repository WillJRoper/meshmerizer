/**
 * @file faces.hpp
 * @brief QEF vertex solving for active octree leaf cells.
 *
 * Given an octree whose active leaf cells each carry corner density
 * values, this module solves the Quadric Error Function (QEF) for
 * each leaf that straddles the isosurface, producing a representative
 * mesh vertex (position + normal).
 *
 * Triangle connectivity is no longer generated here — the pipeline
 * now delegates surface reconstruction to Poisson (via Open3D) in
 * Python, which produces watertight meshes from the oriented point
 * cloud that this module outputs.
 *
 * Design for parallelism:
 * - Vertex solving is per-leaf and independent, suitable for parallel
 *   for-each (OpenMP parallel-for with dynamic scheduling).
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_FACES_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_FACES_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hermite.hpp"
#include "mesh.hpp"
#include "octree_cell.hpp"
#include "omp_config.hpp"
#include "progress_bar.hpp"
#include "qef.hpp"
#include "vector3d.hpp"

/**
 * @brief Solve QEF vertices for all active leaf cells and assign indices.
 *
 * This function iterates over all cells, identifies active leaf cells
 * (leaves with a sign-changing surface), solves the QEF for each, and
 * stores the resulting mesh vertex.  Each active leaf's
 * ``representative_vertex_index`` is set to the index into the returned
 * vertex array.
 *
 * A leaf is considered active if it is a leaf with ``has_surface`` set
 * (meaning its corner values straddle the isovalue).
 *
 * @param all_cells All octree cells (modified: ``representative_vertex_index``
 *     is set for active leaves).
 * @param all_contributors Flat contributor index array.
 * @param positions Particle positions.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue Target surface level.
 * @return Array of mesh vertices, one per active leaf.
 */
inline std::vector<MeshVertex> solve_all_leaf_vertices(
    std::vector<OctreeCell> &all_cells,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue) {
    const std::size_t n_cells = all_cells.size();

    // Per-cell results computed in parallel.  Each entry is either a
    // valid MeshVertex (when the cell is an active surface leaf) or
    // an empty placeholder.  A separate flag marks which cells are
    // active so the serial index-assignment pass can skip non-active
    // cells cheaply.
    std::vector<MeshVertex> per_cell_vertex(n_cells);
    // Use char instead of bool to avoid the std::vector<bool> bit-packing
    // specialisation, which causes data races when adjacent elements are
    // written by different threads (they share the same byte).
    std::vector<char> is_active_leaf(n_cells, 0);

    ProgressBar vertex_bar("Solving vertices", n_cells);

    // Pass 1 (parallel): For each leaf cell, lazily sample corners if
    // needed, compute Hermite samples, and solve the QEF.  Each
    // iteration touches only its own cell and reads shared immutable
    // data (positions, smoothing_lengths, all_contributors), so there
    // are no data races.
#pragma omp parallel for schedule(dynamic)
    for (std::size_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf) {
            vertex_bar.tick();
            continue;
        }

        // Gather contributor indices for this cell.  Construct the
        // vector directly from the iterator range into
        // all_contributors, avoiding per-element push_back overhead.
        // Validate the range to prevent out-of-bounds access from
        // corrupt cell data.
        std::vector<std::size_t> contributors;
        if (cell.contributor_begin >= 0 &&
            cell.contributor_end > cell.contributor_begin) {
            const auto begin_idx =
                static_cast<std::size_t>(cell.contributor_begin);
            const auto end_idx =
                static_cast<std::size_t>(cell.contributor_end);
            if (end_idx > all_contributors.size()) {
                // Cannot throw from an OpenMP region on all
                // compilers; mark the cell inactive and continue.
                vertex_bar.tick();
                continue;
            }
            const auto begin_it =
                all_contributors.begin() +
                static_cast<std::ptrdiff_t>(begin_idx);
            const auto end_it =
                all_contributors.begin() +
                static_cast<std::ptrdiff_t>(end_idx);
            contributors.assign(begin_it, end_it);
        }

        // If corner values were not sampled during refinement (e.g.,
        // the cell was at max depth or had too few contributors at
        // refinement time), sample them now so we can determine
        // whether this leaf actually straddles the isosurface.
        if (!cell.has_surface && !contributors.empty()) {
            cell.corner_values = sample_cell_corners(
                cell, contributors, positions, smoothing_lengths);
            cell.corner_sign_mask = compute_corner_sign_mask(
                cell.corner_values, isovalue);
            cell.has_surface = cell_may_contain_isosurface(
                cell.corner_values, isovalue);
        }

        if (!cell.has_surface) {
            vertex_bar.tick();
            continue;
        }

        // If no sign change, skip (safety check).  The corner_sign_mask
        // was already set during refinement (octree_cell.hpp) or in the
        // lazy sampling block above, so no recomputation is needed here.
        if (cell.corner_sign_mask == 0U ||
            cell.corner_sign_mask == 0xFFU) {
            cell.has_surface = false;
            vertex_bar.tick();
            continue;
        }

        // Compute Hermite samples for this leaf.
        const std::vector<HermiteSample> samples =
            compute_cell_hermite_samples(
                cell.bounds, cell.corner_values,
                cell.corner_sign_mask, contributors,
                positions, smoothing_lengths, isovalue);

        // Solve the QEF.
        per_cell_vertex[cell_idx] =
            solve_qef_for_leaf(samples, cell.bounds);
        is_active_leaf[cell_idx] = 1;
        vertex_bar.tick();
    }

    vertex_bar.finish();

    // Pass 2 (serial): Assign contiguous vertex indices and build
    // the compact output vertex array.  This must be serial because
    // vertex indices must be dense and deterministic.
    std::vector<MeshVertex> vertices;
    for (std::size_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        OctreeCell &cell = all_cells[cell_idx];
        if (is_active_leaf[cell_idx]) {
            cell.representative_vertex_index =
                static_cast<std::int64_t>(vertices.size());
            cell.is_active = true;
            vertices.push_back(per_cell_vertex[cell_idx]);
        } else {
            cell.representative_vertex_index = -1;
        }
    }

    return vertices;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_FACES_HPP_
