#include "refinement_closure.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <span>

#include "octree_cell.hpp"
#include "refinement_context.hpp"
#include "refinement_work_queue.hpp"

namespace {

struct RefinementResult {
    bool has_surface = false;
    bool should_split = false;
    std::array<double, 8> corner_values = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::uint8_t corner_sign_mask = 0U;
    std::vector<OctreeCell> children;
    std::vector<std::vector<std::size_t>> child_contributors;
};

struct PublishedChildren {
    std::vector<std::size_t> child_indices;
    std::vector<std::size_t> affected_indices;
};

class ClosureStorage {
public:
    explicit ClosureStorage(RefinementContext &context) : context_(context) {}

    std::pair<std::int64_t, std::int64_t> publish_contributors(
        const std::vector<std::size_t> &contributors,
        std::size_t reserve_hint = 0U) {
        if (reserve_hint > 0U) {
            context_.contributors().reserve(
                context_.contributors().size() + reserve_hint);
        }
        const std::int64_t contrib_begin =
            static_cast<std::int64_t>(context_.contributors().size());
        for (std::size_t contributor_index : contributors) {
            context_.contributors().push_back(contributor_index);
        }
        const std::int64_t contrib_end =
            static_cast<std::int64_t>(context_.contributors().size());
        return {contrib_begin, contrib_end};
    }

    std::size_t publish_cell(OctreeCell cell) {
        const std::size_t new_index = context_.cells().size();
        context_.cells().push_back(std::move(cell));
        context_.sync_cell_state_size();
        return new_index;
    }

    void reserve_children(std::size_t child_count) {
        if (child_count == 0U) {
            return;
        }
        context_.cells().reserve(context_.cells().size() + child_count);
    }

    RefinementContext &context() { return context_; }

private:
    RefinementContext &context_;
};

class ClosurePublisher {
public:
    ClosurePublisher(
        ClosureStorage &storage,
        BalanceSpatialHash &hash,
        RefinementWorkQueue &queue,
        const std::vector<Vector3d> &positions,
        const std::vector<double> &smoothing_lengths,
        double isovalue)
        : storage_(storage),
          hash_(hash),
          queue_(queue),
          positions_(positions),
          smoothing_lengths_(smoothing_lengths),
          isovalue_(isovalue) {}

    PublishedChildren publish_children(
        std::size_t parent_index,
        std::uint32_t parent_required_depth,
        const std::vector<OctreeCell> &source_children,
        const std::vector<std::vector<std::size_t>> &source_child_contributors,
        bool compute_balance_samples) {
        PublishedChildren published;
        published.affected_indices.reserve(1U + source_children.size());
        published.affected_indices.push_back(parent_index);
        published.child_indices.reserve(source_children.size());
        storage_.reserve_children(source_children.size());

        for (std::size_t child_index = 0;
             child_index < source_children.size();
             ++child_index) {
            OctreeCell child = source_children[child_index];
            const auto [contrib_begin, contrib_end] =
                storage_.publish_contributors(
                    source_child_contributors[child_index],
                    source_child_contributors[child_index].size());

            child.contributor_begin = contrib_begin;
            child.contributor_end = contrib_end;

            if (compute_balance_samples &&
                !source_child_contributors[child_index].empty()) {
                child.is_leaf = true;
                child.is_active = false;
                child.has_surface = false;
                child.is_topo_surface = false;
                child.corner_values = sample_cell_corners(
                    child,
                    std::span<const std::size_t>(
                        source_child_contributors[child_index]),
                    positions_,
                    smoothing_lengths_);
                child.corner_sign_mask = compute_corner_sign_mask(
                    child.corner_values, isovalue_);
                child.has_surface = cell_may_contain_isosurface(
                    child.corner_values, isovalue_);
            }

            const std::size_t new_index = storage_.publish_cell(child);
            storage_.context().raise_required_depth_to(
                new_index,
                std::max(child.depth, parent_required_depth));
            if (storage_.context().mark_queued(new_index)) {
                queue_.push({
                    new_index,
                    storage_.context().get_required_depth(new_index),
                    0U,
                });
            }

            published.child_indices.push_back(new_index);
            published.affected_indices.push_back(new_index);

            std::uint32_t cgx, cgy, cgz;
            hash_.quantize(child.bounds.min, cgx, cgy, cgz);
            hash_.map[balance_pack_coords(cgx, cgy, cgz)] = new_index;
        }

        return published;
    }

private:
    ClosureStorage &storage_;
    BalanceSpatialHash &hash_;
    RefinementWorkQueue &queue_;
    const std::vector<Vector3d> &positions_;
    const std::vector<double> &smoothing_lengths_;
    double isovalue_;
};

inline RefinementResult evaluate_refinement_for_leaf(
    const OctreeCell &current_cell,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config) {
    RefinementResult result;

    const std::int64_t contrib_begin = current_cell.contributor_begin;
    const std::int64_t contrib_end = current_cell.contributor_end;
    if (contrib_begin < 0 || contrib_end < 0 || contrib_begin >= contrib_end) {
        return result;
    }

    const auto safe_begin = static_cast<std::size_t>(
        std::min(contrib_begin,
                 static_cast<std::int64_t>(all_contributors.size())));
    const auto safe_end = static_cast<std::size_t>(
        std::min(contrib_end,
                 static_cast<std::int64_t>(all_contributors.size())));
    const std::span<const std::size_t> contributors =
        contributor_span(all_contributors, safe_begin, safe_end);

    if (contributors.size() < 2) {
        return result;
    }

    result.corner_values = sample_cell_corners(
        current_cell, contributors, positions, smoothing_lengths);
    result.corner_sign_mask = compute_corner_sign_mask(
        result.corner_values, config.isovalue);

    const bool corner_surface = cell_may_contain_isosurface(
        result.corner_values, config.isovalue);
    const bool inherited_surface_hint =
        current_cell.has_surface && !corner_surface;
    result.has_surface = corner_surface || inherited_surface_hint;
    if (!result.has_surface) {
        return result;
    }

    if (current_cell.depth >= config.max_depth) {
        return result;
    }

    if (!corner_surface) {
        result.should_split = true;
        result.children = create_child_cells(current_cell);
        result.child_contributors = filter_child_contributors(
            contributors, positions, smoothing_lengths, result.children);
        return result;
    }

    const std::vector<HermiteSample> samples = compute_cell_hermite_samples(
        current_cell.bounds, result.corner_values,
        result.corner_sign_mask, contributors,
        positions, smoothing_lengths, config.isovalue);
    const QEFLeafDiagnostics qef_diagnostics =
        analyze_qef_for_leaf(samples, current_cell.bounds);
    const double dx = current_cell.bounds.max.x - current_cell.bounds.min.x;
    const double dy = current_cell.bounds.max.y - current_cell.bounds.min.y;
    const double dz = current_cell.bounds.max.z - current_cell.bounds.min.z;
    const double cell_radius =
        0.5 * std::sqrt(dx * dx + dy * dy + dz * dz);
    const bool poor_qef_fit =
        qef_diagnostics.usable_sample_count <
            config.minimum_usable_hermite_samples ||
        qef_diagnostics.used_fallback ||
        minimum_hermite_normal_alignment(samples) <
            config.min_normal_alignment_threshold ||
        qef_diagnostics.rms_plane_residual >
            config.max_qef_rms_residual_ratio * cell_radius;
    if (!poor_qef_fit) {
        return result;
    }

    result.should_split = true;
    result.children = create_child_cells(current_cell);
    result.child_contributors = filter_child_contributors(
        contributors, positions, smoothing_lengths, result.children);
    for (OctreeCell &child : result.children) {
        child.has_surface = child_inherits_surface_hint(
            child, samples, qef_diagnostics.vertex);
    }

    return result;
}

inline void schedule_balance_neighbors_for_cell(
    std::size_t cell_index,
    const std::vector<OctreeCell> &all_cells,
    const BalanceSpatialHash &hash,
    std::uint32_t max_depth,
    RefinementContext &context,
    RefinementWorkQueue &queue) {
    if (cell_index >= all_cells.size()) {
        return;
    }

    auto enqueue_index = [&](std::size_t candidate, std::uint32_t demanded_depth) {
        if (candidate >= all_cells.size()) {
            return;
        }
        if (!all_cells[candidate].is_leaf) {
            return;
        }
        if (context.get_required_depth(candidate) >= demanded_depth) {
            return;
        }
        context.raise_required_depth_to(candidate, demanded_depth);
        if (context.mark_queued(candidate)) {
            queue.push({
                candidate,
                context.get_required_depth(candidate),
                0U,
            });
        }
    };

    const OctreeCell &cell = all_cells[cell_index];
    const std::uint32_t demanded_depth = cell.depth > 0U ? cell.depth - 1U : 0U;
    enqueue_index(cell_index, demanded_depth);
    if (!cell.is_leaf) {
        return;
    }

    const std::uint32_t span = 1U << (max_depth - cell.depth);
    std::uint32_t gx, gy, gz;
    hash.quantize(cell.bounds.min, gx, gy, gz);
    std::set<std::tuple<int, bool, std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t>> visited_patches;

    auto recurse_face_patch = [&](auto &&self,
                                  int axis,
                                  bool positive_direction,
                                  std::uint32_t patch_u_min,
                                  std::uint32_t patch_v_min,
                                  std::uint32_t patch_u_size,
                                  std::uint32_t patch_v_size) -> void {
        if (patch_u_size == 0U || patch_v_size == 0U) {
            return;
        }
        const auto patch_key = std::make_tuple(
            axis,
            positive_direction,
            patch_u_min,
            patch_v_min,
            patch_u_size,
            patch_v_size);
        if (!visited_patches.insert(patch_key).second) {
            return;
        }

        const int u_axis = axis == 0 ? 1 : 0;
        const int v_axis = axis == 2 ? 1 : 2;
        const std::int64_t face_coord =
            positive_direction ?
            static_cast<std::int64_t>((axis == 0 ? gx : axis == 1 ? gy : gz) + span) :
            static_cast<std::int64_t>((axis == 0 ? gx : axis == 1 ? gy : gz) - 1);

        std::array<std::int64_t, 3> sample = {
            static_cast<std::int64_t>(gx),
            static_cast<std::int64_t>(gy),
            static_cast<std::int64_t>(gz),
        };
        sample[axis] = face_coord;
        sample[u_axis] += static_cast<std::int64_t>(patch_u_min);
        sample[v_axis] += static_cast<std::int64_t>(patch_v_min);
        if (sample[0] < 0 || sample[1] < 0 || sample[2] < 0) {
            return;
        }

        const std::size_t neighbor_idx = hash.find_leaf_at(
            static_cast<std::uint32_t>(sample[0]),
            static_cast<std::uint32_t>(sample[1]),
            static_cast<std::uint32_t>(sample[2]));
        if (neighbor_idx == SIZE_MAX) {
            return;
        }

        const OctreeCell &neighbor = all_cells[neighbor_idx];
        if (neighbor.depth < demanded_depth) {
            enqueue_index(neighbor_idx, demanded_depth);
        }

        std::uint32_t ngx, ngy, ngz;
        hash.quantize(neighbor.bounds.min, ngx, ngy, ngz);
        const std::uint32_t neighbor_span = 1U << (max_depth - neighbor.depth);
        const std::array<std::uint32_t, 3> neighbor_min = {ngx, ngy, ngz};
        const std::array<std::uint32_t, 3> neighbor_max = {
            ngx + neighbor_span,
            ngy + neighbor_span,
            ngz + neighbor_span,
        };
        const std::uint32_t patch_u_max = patch_u_min + patch_u_size;
        const std::uint32_t patch_v_max = patch_v_min + patch_v_size;
        const std::uint32_t abs_patch_u_min =
            (u_axis == 0 ? gx : u_axis == 1 ? gy : gz) + patch_u_min;
        const std::uint32_t abs_patch_u_max =
            (u_axis == 0 ? gx : u_axis == 1 ? gy : gz) + patch_u_max;
        const std::uint32_t abs_patch_v_min =
            (v_axis == 0 ? gx : v_axis == 1 ? gy : gz) + patch_v_min;
        const std::uint32_t abs_patch_v_max =
            (v_axis == 0 ? gx : v_axis == 1 ? gy : gz) + patch_v_max;
        const bool leaf_covers_patch =
            neighbor_min[u_axis] <= abs_patch_u_min &&
            abs_patch_u_max <= neighbor_max[u_axis] &&
            neighbor_min[v_axis] <= abs_patch_v_min &&
            abs_patch_v_max <= neighbor_max[v_axis];

        if (leaf_covers_patch || (patch_u_size == 1U && patch_v_size == 1U)) {
            return;
        }

        const std::uint32_t u_half = patch_u_size > 1U ? patch_u_size / 2U : 0U;
        const std::uint32_t v_half = patch_v_size > 1U ? patch_v_size / 2U : 0U;
        const std::uint32_t u_sizes[2] = {u_half, patch_u_size - u_half};
        const std::uint32_t v_sizes[2] = {v_half, patch_v_size - v_half};
        const std::uint32_t u_offsets[2] = {patch_u_min, patch_u_min + u_half};
        const std::uint32_t v_offsets[2] = {patch_v_min, patch_v_min + v_half};

        for (int ui = 0; ui < 2; ++ui) {
            for (int vi = 0; vi < 2; ++vi) {
                if (u_sizes[ui] == 0U || v_sizes[vi] == 0U) {
                    continue;
                }
                if (u_sizes[ui] == patch_u_size && v_sizes[vi] == patch_v_size) {
                    continue;
                }
                self(
                    self,
                    axis,
                    positive_direction,
                    u_offsets[ui],
                    v_offsets[vi],
                    u_sizes[ui],
                    v_sizes[vi]);
            }
        }
    };

    recurse_face_patch(recurse_face_patch, 0, true, 0U, 0U, span, span);
    recurse_face_patch(recurse_face_patch, 0, false, 0U, 0U, span, span);
    recurse_face_patch(recurse_face_patch, 1, true, 0U, 0U, span, span);
    recurse_face_patch(recurse_face_patch, 1, false, 0U, 0U, span, span);
    recurse_face_patch(recurse_face_patch, 2, true, 0U, 0U, span, span);
    recurse_face_patch(recurse_face_patch, 2, false, 0U, 0U, span, span);
}

inline void apply_balance_split(
    std::size_t split_index,
    RefinementContext &context,
    BalanceSpatialHash &hash,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    RefinementWorkQueue &queue) {
    OctreeCell &current_cell = context.cells()[split_index];
    const OctreeCell parent_snapshot = current_cell;
    const std::uint32_t parent_required_depth =
        context.get_required_depth(split_index);
    const std::span<const std::size_t> parent_contributors = contributor_span(
        context.contributors(),
        parent_snapshot.contributor_begin < 0 ? 0U :
            static_cast<std::size_t>(parent_snapshot.contributor_begin),
        parent_snapshot.contributor_end <= parent_snapshot.contributor_begin ? 0U :
            static_cast<std::size_t>(parent_snapshot.contributor_end));

    const std::vector<OctreeCell> children = create_child_cells(parent_snapshot);
    const std::vector<std::vector<std::size_t>> child_contributors =
        filter_child_contributors(
            parent_contributors, positions, smoothing_lengths, children);

    current_cell.is_leaf = false;
    current_cell.child_begin = static_cast<std::int64_t>(context.cells().size());

    std::uint32_t pgx, pgy, pgz;
    hash.quantize(parent_snapshot.bounds.min, pgx, pgy, pgz);
    hash.map.erase(balance_pack_coords(pgx, pgy, pgz));

    ClosureStorage storage(context);
    ClosurePublisher publisher(
        storage, hash, queue, positions, smoothing_lengths, config.isovalue);

    const PublishedChildren published =
        publisher.publish_children(
            split_index,
            parent_required_depth,
            children,
            child_contributors,
            true);

    for (std::size_t affected_index : published.affected_indices) {
        schedule_balance_neighbors_for_cell(
            affected_index,
            context.cells(),
            hash,
            config.max_depth,
            context,
            queue);
    }
}

inline void apply_surface_split(
    std::size_t split_index,
    const RefinementResult &result,
    RefinementContext &context,
    BalanceSpatialHash &hash,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    RefinementWorkQueue &queue) {
    OctreeCell &current_cell = context.cells()[split_index];
    const OctreeCell parent_snapshot = current_cell;
    const std::uint32_t parent_required_depth =
        context.get_required_depth(split_index);

    current_cell.is_leaf = false;
    current_cell.is_active = true;
    current_cell.has_surface = true;
    current_cell.is_topo_surface = false;
    current_cell.child_begin = static_cast<std::int64_t>(context.cells().size());

    std::uint32_t pgx, pgy, pgz;
    hash.quantize(parent_snapshot.bounds.min, pgx, pgy, pgz);
    hash.map.erase(balance_pack_coords(pgx, pgy, pgz));

    ClosureStorage storage(context);
    ClosurePublisher publisher(
        storage, hash, queue, positions, smoothing_lengths, config.isovalue);

    const PublishedChildren published =
        publisher.publish_children(
            split_index,
            parent_required_depth,
            result.children,
            result.child_contributors,
            false);

    for (std::size_t affected_index : published.affected_indices) {
        schedule_balance_neighbors_for_cell(
            affected_index,
            context.cells(),
            hash,
            config.max_depth,
            context,
            queue);
    }
}

}  // namespace

std::pair<std::vector<OctreeCell>, std::vector<std::size_t>>
refine_with_closure(
    std::vector<OctreeCell> initial_cells,
    std::vector<std::size_t> initial_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config) {
    if (initial_cells.empty()) {
        return {{}, {}};
    }

    RefinementContext context(initial_cells, initial_contributors);
    context.sync_cell_state_size();

    BalanceSpatialHash hash;
    hash.build(
        context.cells(), config.domain, config.max_depth,
        config.base_resolution);

    RefinementWorkQueue queue;
    for (std::size_t cell_index = 0; cell_index < context.size(); ++cell_index) {
        context.raise_required_depth_to(
            cell_index,
            initial_cells[cell_index].depth);
        if (context.mark_queued(cell_index)) {
            queue.push({
                cell_index,
                context.get_required_depth(cell_index),
                0U,
            });
        }
    }

    RefinementTask task;
    while (queue.pop(task)) {
        if (!context.mark_processing(task.cell_index)) {
            continue;
        }
        if (task.cell_index >= context.cells().size()) {
            continue;
        }

        OctreeCell &current_cell = context.cells()[task.cell_index];
        if (!current_cell.is_leaf) {
            context.mark_retired(task.cell_index);
            if (queue.empty()) {
                queue.shutdown();
            }
            continue;
        }

        if (current_cell.depth < context.get_required_depth(task.cell_index)) {
            apply_balance_split(
                task.cell_index,
                context,
                hash,
                positions,
                smoothing_lengths,
                config,
                queue);
            context.mark_retired(task.cell_index);
            if (queue.empty()) {
                queue.shutdown();
            }
            continue;
        }

        const RefinementResult result = evaluate_refinement_for_leaf(
            current_cell,
            context.contributors(),
            positions,
            smoothing_lengths,
            config);

        current_cell.corner_values = result.corner_values;
        current_cell.corner_sign_mask = result.corner_sign_mask;

        if (!result.has_surface) {
            current_cell.is_leaf = true;
            current_cell.is_active = false;
            current_cell.has_surface = false;
            current_cell.is_topo_surface = false;
            current_cell.child_begin = -1;
            context.mark_idle(task.cell_index);
            if (queue.empty()) {
                queue.shutdown();
            }
            continue;
        }

        if (!result.should_split) {
            current_cell.is_leaf = true;
            current_cell.is_active = true;
            current_cell.has_surface = true;
            current_cell.is_topo_surface = false;
            current_cell.child_begin = -1;
            context.mark_idle(task.cell_index);
            if (queue.empty()) {
                queue.shutdown();
            }
            continue;
        }

        apply_surface_split(
            task.cell_index,
            result,
            context,
            hash,
            positions,
            smoothing_lengths,
            config,
            queue);

        context.mark_retired(task.cell_index);

        if (queue.empty()) {
            queue.shutdown();
        }
    }

    return {
        std::move(initial_cells),
        std::move(initial_contributors),
    };
}
