/**
 * @file fof.hpp
 * @brief Friends-of-friends clustering for oriented point clouds.
 *
 * Given a set of 3-D positions (typically QEF vertex positions from the
 * adaptive octree), this module groups them into connected components
 * using a friends-of-friends (FOF) algorithm with a spatial-hash
 * acceleration structure.
 *
 * The linking length is computed automatically from the mean inter-point
 * separation measured over the tight bounding box of the input points:
 *
 *     mean_sep = (tight_bbox_volume / n_points) ^ (1/3)
 *     linking_length = linking_factor * mean_sep
 *
 * where ``linking_factor`` is a tunable parameter (default 1.5).
 *
 * Algorithm:
 *
 * 1. Build a spatial hash with bin size equal to the linking length.
 *    Each point is assigned to a bin based on its integer grid
 *    coordinates ``(floor((x - x_min) / linking_length), ...)``.
 *
 * 2. For each point, check the 27 neighboring bins (3x3x3 block
 *    centered on the point's own bin).  For every point in a
 *    neighboring bin, if the Euclidean distance is less than the
 *    linking length, union the two points in a disjoint-set structure.
 *
 * 3. Flatten the disjoint-set forest so every point's label is the
 *    root of its component.  Relabel roots to contiguous integers
 *    starting from 0.
 *
 * Complexity: O(N * K) where K is the average number of points per
 * 27-bin neighborhood (typically small and bounded by the linking
 * length choice).  The Union-Find operations are amortized O(alpha(N))
 * per union/find thanks to path compression and union by rank.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_FOF_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_FOF_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "cancellation.hpp"
#include "progress_bar.hpp"
#include "vector3d.hpp"

/**
 * @brief Disjoint-set (Union-Find) with path compression and union by rank.
 *
 * Each element starts as its own set.  ``unite(a, b)`` merges the sets
 * containing ``a`` and ``b``.  ``find(a)`` returns the canonical root
 * of the set containing ``a``, compressing the path along the way.
 *
 * The amortized cost per operation is O(alpha(N)) where alpha is the
 * inverse Ackermann function — effectively constant for all practical
 * input sizes.
 */
struct DisjointSet {
    /** @brief Parent pointers.  parent[i] == i means i is a root. */
    std::vector<std::size_t> parent;

    /** @brief Rank (upper bound on tree height) for union by rank. */
    std::vector<std::size_t> rank;

    /**
     * @brief Initialize N singleton sets.
     *
     * @param n Number of elements (each starts as its own set).
     */
    void init(std::size_t n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    /**
     * @brief Find the root of the set containing element ``x``.
     *
     * Uses path compression: every node along the path from ``x`` to
     * the root is re-pointed directly to the root, flattening the
     * tree for future queries.
     *
     * @param x Element index.
     * @return Root index of the set containing ``x``.
     */
    std::size_t find(std::size_t x) {
        while (parent[x] != x) {
            // Path splitting: point x to its grandparent.
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    /**
     * @brief Merge the sets containing elements ``a`` and ``b``.
     *
     * Uses union by rank: the shorter tree is attached under the
     * taller tree's root, keeping the tree balanced.
     *
     * @param a First element index.
     * @param b Second element index.
     */
    void unite(std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        // Attach the shorter tree under the taller tree.
        if (rank[a] < rank[b]) {
            parent[a] = b;
        } else if (rank[a] > rank[b]) {
            parent[b] = a;
        } else {
            parent[b] = a;
            ++rank[a];
        }
    }
};

/**
 * @brief Hash key for a 3-D integer grid cell.
 *
 * Packs three 32-bit grid coordinates into a single 96-bit key
 * (stored as a struct for hashing).  The coordinates can be negative
 * (signed) to handle points near the domain boundary.
 */
struct FofGridKey {
    std::int32_t ix;
    std::int32_t iy;
    std::int32_t iz;

    bool operator==(const FofGridKey &other) const {
        return ix == other.ix && iy == other.iy && iz == other.iz;
    }
};

/**
 * @brief Hash function for FofGridKey.
 *
 * Uses a simple multiplicative hash combining the three coordinates.
 */
struct FofGridKeyHash {
    std::size_t operator()(const FofGridKey &key) const {
        // Combine the three coordinates with large primes to reduce
        // collisions.  The constants are arbitrary odd primes.
        std::size_t h = static_cast<std::size_t>(
            static_cast<std::uint32_t>(key.ix));
        h ^= static_cast<std::size_t>(
                 static_cast<std::uint32_t>(key.iy)) *
             2654435761ULL;
        h ^= static_cast<std::size_t>(
                 static_cast<std::uint32_t>(key.iz)) *
             40503ULL;
        // Final mix.
        h ^= h >> 16U;
        h *= 0x45d9f3bULL;
        h ^= h >> 16U;
        return h;
    }
};

/**
 * @brief Run friends-of-friends clustering on a set of 3-D positions.
 *
 * Computes the linking length from the domain volume and point count,
 * builds a spatial hash, and uses Union-Find to merge points within
 * the linking length.  Returns a per-point group label array with
 * contiguous labels starting from 0.
 *
 * @param positions     Array of N 3-D positions (QEF vertices).
 * @param domain_min    Minimum corner of the nominal domain bounding box.
 * @param domain_max    Maximum corner of the nominal domain bounding box.
 * @param linking_factor Multiplier for the mean inter-point separation.
 *     Default 1.5.  Larger values merge more aggressively.
 * @return Vector of N group labels (contiguous integers from 0).
 */
inline std::vector<std::int64_t> fof_cluster(
    const std::vector<Vector3d> &positions,
    const Vector3d &domain_min,
    const Vector3d &domain_max,
    double linking_factor = 1.5) {
    const std::size_t n = positions.size();

    // Handle empty input.
    if (n == 0) {
        return {};
    }

    // Handle single point.
    if (n == 1) {
        return {0};
    }

    // Compute a tight bounding box over the occupied points. Using the
    // full selected domain can dramatically overestimate the mean
    // separation when the particles occupy only a small sub-volume,
    // which in turn makes the linking length far too large and slows
    // the clustering step.
    Vector3d tight_min = positions[0];
    Vector3d tight_max = positions[0];
    for (std::size_t i = 1; i < n; ++i) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(i);
        tight_min.x = std::min(tight_min.x, positions[i].x);
        tight_min.y = std::min(tight_min.y, positions[i].y);
        tight_min.z = std::min(tight_min.z, positions[i].z);
        tight_max.x = std::max(tight_max.x, positions[i].x);
        tight_max.y = std::max(tight_max.y, positions[i].y);
        tight_max.z = std::max(tight_max.z, positions[i].z);
    }

    const double extent_x = std::max(tight_max.x - tight_min.x, 0.0);
    const double extent_y = std::max(tight_max.y - tight_min.y, 0.0);
    const double extent_z = std::max(tight_max.z - tight_min.z, 0.0);
    double occupied_vol = extent_x * extent_y * extent_z;

    // If the occupied extent is degenerate along one or more axes,
    // fall back to the nominal domain bounds so the linking length
    // remains finite and the hash still has a sensible scale.
    if (!(occupied_vol > 0.0)) {
        occupied_vol =
            std::max(domain_max.x - domain_min.x, 0.0) *
            std::max(domain_max.y - domain_min.y, 0.0) *
            std::max(domain_max.z - domain_min.z, 0.0);
    }
    if (!(occupied_vol > 0.0)) {
        occupied_vol = 1.0;
    }

    const double mean_sep = std::cbrt(occupied_vol / static_cast<double>(n));
    const double linking_length = linking_factor * mean_sep;
    const double linking_length_sq = linking_length * linking_length;

    // Inverse linking length for bin coordinate computation.
    const double inv_ll = 1.0 / linking_length;

    // Build spatial hash: map each grid cell to the list of point
    // indices that fall within it.
    std::unordered_map<FofGridKey, std::vector<std::size_t>, FofGridKeyHash>
        grid;
    // Pre-compute bin coordinates for each point and insert.
    std::vector<FofGridKey> point_bins(n);
    for (std::size_t i = 0; i < n; ++i) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(i);
        const std::int32_t bx = static_cast<std::int32_t>(
            std::floor((positions[i].x - tight_min.x) * inv_ll));
        const std::int32_t by = static_cast<std::int32_t>(
            std::floor((positions[i].y - tight_min.y) * inv_ll));
        const std::int32_t bz = static_cast<std::int32_t>(
            std::floor((positions[i].z - tight_min.z) * inv_ll));
        point_bins[i] = {bx, by, bz};
        grid[point_bins[i]].push_back(i);
    }

    // Initialize Union-Find.
    DisjointSet dset;
    dset.init(n);

    ProgressBar fof_bar("Clustering", "fof_cluster", n);

    // For each point, check the 27 neighboring bins and union with
    // any point within the linking length.
    for (std::size_t i = 0; i < n; ++i) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(i);
        const FofGridKey &bin_i = point_bins[i];
        const Vector3d &pos_i = positions[i];

        // Iterate over the 3x3x3 neighborhood of bins.
        for (std::int32_t dx = -1; dx <= 1; ++dx) {
            for (std::int32_t dy = -1; dy <= 1; ++dy) {
                for (std::int32_t dz = -1; dz <= 1; ++dz) {
                    const FofGridKey neighbor_key = {
                        bin_i.ix + dx,
                        bin_i.iy + dy,
                        bin_i.iz + dz,
                    };
                    auto it = grid.find(neighbor_key);
                    if (it == grid.end()) {
                        continue;
                    }
                    for (std::size_t j : it->second) {
                        // Skip self and already-linked pairs.
                        // The find() check avoids redundant distance
                        // computations for points already in the same
                        // group.
                        if (j <= i) {
                            continue;
                        }
                        if (dset.find(i) == dset.find(j)) {
                            continue;
                        }
                        const double ddx =
                            pos_i.x - positions[j].x;
                        const double ddy =
                            pos_i.y - positions[j].y;
                        const double ddz =
                            pos_i.z - positions[j].z;
                        const double dist_sq =
                            ddx * ddx + ddy * ddy + ddz * ddz;
                        if (dist_sq < linking_length_sq) {
                            dset.unite(i, j);
                        }
                    }
                }
            }
        }
        fof_bar.tick();
    }

    fof_bar.finish();

    // Flatten the disjoint-set forest and relabel roots to contiguous
    // integers starting from 0.
    std::unordered_map<std::size_t, std::int64_t> root_to_label;
    std::vector<std::int64_t> labels(n);
    std::int64_t next_label = 0;
    for (std::size_t i = 0; i < n; ++i) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(i);
        const std::size_t root = dset.find(i);
        auto it = root_to_label.find(root);
        if (it == root_to_label.end()) {
            root_to_label[root] = next_label;
            labels[i] = next_label;
            ++next_label;
        } else {
            labels[i] = it->second;
        }
    }

    return labels;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_FOF_HPP_
