/**
 * @file particle_grid.hpp
 * @brief Top-level particle binning helpers for adaptive contributor queries.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_PARTICLE_GRID_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_PARTICLE_GRID_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "progress_bar.hpp"
#include "bounding_box.hpp"
#include "omp_config.hpp"

/**
 * @brief One uniform top-level bin storing particle indices.
 *
 * The adaptive rewrite uses top-level bins as the first spatial acceleration
 * structure so contributor attachment can query nearby particles without a full
 * snapshot scan for every cell.
 */
struct TopLevelBin {
    std::vector<std::size_t> particle_indices;
};

/**
 * @brief Uniform top-level particle grid over the working domain.
 */
struct TopLevelParticleGrid {
    BoundingBox domain;
    std::uint32_t resolution;
    std::vector<TopLevelBin> bins;

    /** Cached reciprocal of bin size for fast coordinate mapping. */
    Vector3d inv_bin_size_;

    /**
     * @brief Construct the grid and allocate all bins.
     *
     * @param input_domain Working domain covered by the top-level grid.
     * @param input_resolution Number of bins along each axis.
     */
    TopLevelParticleGrid(const BoundingBox &input_domain,
                         std::uint32_t input_resolution)
        : domain(input_domain), resolution(input_resolution),
          bins(static_cast<std::size_t>(resolution) * resolution * resolution) {
        if (resolution == 0U) {
            throw std::invalid_argument(
                "TopLevelParticleGrid resolution must be > 0");
        }
        // Pre-compute inverse bin sizes so bin_coordinates can multiply
        // instead of dividing on every call.
        const double res_d = static_cast<double>(resolution);
        inv_bin_size_ = {
            res_d / (domain.max.x - domain.min.x),
            res_d / (domain.max.y - domain.min.y),
            res_d / (domain.max.z - domain.min.z),
        };
    }

    /**
     * @brief Return the world-space width of one top-level bin.
     *
     * @return Bin width along each axis.
     */
    Vector3d bin_size() const {
        return {
            (domain.max.x - domain.min.x) / static_cast<double>(resolution),
            (domain.max.y - domain.min.y) / static_cast<double>(resolution),
            (domain.max.z - domain.min.z) / static_cast<double>(resolution),
        };
    }

    /**
     * @brief Return the flattened bin index for one integer bin coordinate.
     *
     * @param ix X-axis bin coordinate.
     * @param iy Y-axis bin coordinate.
     * @param iz Z-axis bin coordinate.
     * @return Flattened row-major bin index.
     */
    std::size_t flatten_index(std::uint32_t ix, std::uint32_t iy,
                              std::uint32_t iz) const {
        return (static_cast<std::size_t>(ix) * resolution + iy) * resolution +
               iz;
    }

    /**
     * @brief Return the bin coordinates for one point.
     *
     * Points outside the domain are clamped to the nearest boundary bin.
     *
     * @param point World-space point.
     * @return Integer bin coordinates.
     */
    Vector3d bin_coordinates(const Vector3d &point) const {
        const auto clamp_axis = [this](double value) {
            return std::max(
                0.0,
                std::min(value, static_cast<double>(resolution - 1)));
        };
        return {
            clamp_axis((point.x - domain.min.x) * inv_bin_size_.x),
            clamp_axis((point.y - domain.min.y) * inv_bin_size_.y),
            clamp_axis((point.z - domain.min.z) * inv_bin_size_.z),
        };
    }

    /**
     * @brief Insert particles into their owning top-level bins.
     *
     * When compiled with OpenMP (``WITH_OPENMP`` defined), uses a
     * three-pass parallel algorithm:
     *
     *   1. **Count pass** (parallel): compute the bin index for each
     *      particle and atomically increment per-bin counts.
     *   2. **Prefix-sum pass** (serial): convert counts to write
     *      offsets via an exclusive prefix sum.
     *   3. **Scatter pass** (parallel): write each particle's index
     *      into the correct position in each bin's pre-allocated
     *      storage using an atomic fetch-and-add on a per-bin write
     *      cursor.  Each slot is written exactly once, so there is
     *      no data race on the storage itself.
     *
     * Without OpenMP, falls back to the original serial loop.
     *
     * Note: the parallel path produces bins with identical contents
     * to the serial path, but the order of particle indices within
     * each bin may differ.  This is acceptable because no downstream
     * code depends on intra-bin ordering.
     *
     * @param positions Particle positions in world space.
     */
    void insert_particles(const std::vector<Vector3d> &positions) {
        ProgressBar grid_bar("Binning particles",
                             positions.size());
#ifdef WITH_OPENMP
        const std::size_t n_particles = positions.size();
        const std::size_t n_bins = bins.size();

        // Per-particle bin assignment computed in pass 1.
        std::vector<std::size_t> particle_bin(n_particles);

        // Per-bin counts (use int for OpenMP atomic compatibility).
        std::vector<int> bin_counts(n_bins, 0);

        // Pass 1: compute bin index per particle and count per bin.
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n_particles; ++i) {
            const Vector3d coords = bin_coordinates(positions[i]);
            const std::size_t bin_idx = flatten_index(
                static_cast<std::uint32_t>(coords.x),
                static_cast<std::uint32_t>(coords.y),
                static_cast<std::uint32_t>(coords.z));
            particle_bin[i] = bin_idx;
#pragma omp atomic
            bin_counts[bin_idx]++;
        }

        // Pass 2: pre-allocate each bin and set up write cursors.
        // Also create a per-bin atomic write cursor initialised to 0.
        std::vector<int> bin_cursor(n_bins, 0);
        for (std::size_t b = 0; b < n_bins; ++b) {
            bins[b].particle_indices.resize(
                static_cast<std::size_t>(bin_counts[b]));
        }

        // Pass 3: scatter particle indices into pre-allocated bins.
        // Each bin slot is written exactly once (cursor increments
        // atomically), so there is no race on the storage vector's
        // data buffer.
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n_particles; ++i) {
            const std::size_t bin_idx = particle_bin[i];
            int pos;
#pragma omp atomic capture
            pos = bin_cursor[bin_idx]++;
            bins[bin_idx].particle_indices[
                static_cast<std::size_t>(pos)] = i;
            grid_bar.tick();
        }
#else
        // Serial fallback: simple loop with push_back.
        for (std::size_t particle_index = 0;
             particle_index < positions.size();
             ++particle_index) {
            const Vector3d coords =
                bin_coordinates(positions[particle_index]);
            bins[flatten_index(
                     static_cast<std::uint32_t>(coords.x),
                     static_cast<std::uint32_t>(coords.y),
                     static_cast<std::uint32_t>(coords.z))]
                .particle_indices.push_back(particle_index);
            grid_bar.tick();
        }
#endif
        grid_bar.finish();
    }

    /**
     * @brief Return the integer bin span overlapped by a bounding box.
     *
     * @param box Query bounding box.
     * @param start Output inclusive start coordinates.
     * @param stop Output inclusive stop coordinates.
     */
    void overlapping_bin_span(const BoundingBox &box, std::uint32_t &start_x,
                              std::uint32_t &start_y, std::uint32_t &start_z,
                              std::uint32_t &stop_x, std::uint32_t &stop_y,
                              std::uint32_t &stop_z) const {
        const Vector3d start = bin_coordinates(box.min);
        const Vector3d stop = bin_coordinates(
            {
                std::nextafter(box.max.x, box.min.x),
                std::nextafter(box.max.y, box.min.y),
                std::nextafter(box.max.z, box.min.z),
            });
        start_x = static_cast<std::uint32_t>(start.x);
        start_y = static_cast<std::uint32_t>(start.y);
        start_z = static_cast<std::uint32_t>(start.z);
        stop_x = static_cast<std::uint32_t>(stop.x);
        stop_y = static_cast<std::uint32_t>(stop.y);
        stop_z = static_cast<std::uint32_t>(stop.z);
    }

    /**
     * @brief Return the bin span that could contain overlapping contributors.
     *
     * The query box is expanded by a support padding so particles whose centers
     * lie outside the cell's own bins can still be considered if their kernels
     * reach into the cell.
     *
     * @param box Query bounding box.
     * @param padding Maximum support radius to include around the box.
     * @param start_x Output inclusive start x-coordinate.
     * @param start_y Output inclusive start y-coordinate.
     * @param start_z Output inclusive start z-coordinate.
     * @param stop_x Output inclusive stop x-coordinate.
     * @param stop_y Output inclusive stop y-coordinate.
     * @param stop_z Output inclusive stop z-coordinate.
     */
    void contributor_bin_span(const BoundingBox &box, double padding,
                              std::uint32_t &start_x, std::uint32_t &start_y,
                              std::uint32_t &start_z, std::uint32_t &stop_x,
                              std::uint32_t &stop_y,
                              std::uint32_t &stop_z) const {
        const BoundingBox expanded = {
            {
                box.min.x - padding,
                box.min.y - padding,
                box.min.z - padding,
            },
            {
                box.max.x + padding,
                box.max.y + padding,
                box.max.z + padding,
            },
        };
        overlapping_bin_span(expanded, start_x, start_y, start_z, stop_x,
                             stop_y, stop_z);
    }
};

/**
 * @brief Return whether a particle support sphere overlaps a cell box.
 *
 * @param position Particle center.
 * @param smoothing_length Particle support radius.
 * @param cell Bounding box for the candidate cell.
 * @return True when the support sphere overlaps the cell box.
 */
inline bool particle_support_overlaps_box(const Vector3d &position,
                                          double smoothing_length,
                                          const BoundingBox &cell) {
    const auto clamp = [](double value, double lower, double upper) {
        return std::max(lower, std::min(value, upper));
    };
    const double px = clamp(position.x, cell.min.x, cell.max.x);
    const double py = clamp(position.y, cell.min.y, cell.max.y);
    const double pz = clamp(position.z, cell.min.z, cell.max.z);
    const double dx = position.x - px;
    const double dy = position.y - py;
    const double dz = position.z - pz;
    return dx * dx + dy * dy + dz * dz <=
           smoothing_length * smoothing_length;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_PARTICLE_GRID_HPP_
