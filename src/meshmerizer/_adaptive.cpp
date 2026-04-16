/**
 * @file _adaptive.cpp
 * @brief Minimal C++ extension scaffold for the adaptive meshing rewrite.
 *
 * This file intentionally exposes only a tiny Python API at the start of the
 * rewrite. Its purpose is to prove that the repository can build and import a
 * dedicated C++ extension for the adaptive meshing path before larger classes
 * and algorithms are added.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <vector>

#include "adaptive_cpp/bounding_box.hpp"
#include "adaptive_cpp/kernel_wendland_c2.hpp"
#include "adaptive_cpp/morton.hpp"
#include "adaptive_cpp/octree_cell.hpp"
#include "adaptive_cpp/particle.hpp"
#include "adaptive_cpp/particle_grid.hpp"

/**
 * @brief Parse a Python `(x, y, z)` tuple into a `Vector3d`.
 *
 * @param object Python tuple-like object.
 * @param output Parsed vector destination.
 * @return `true` when parsing succeeds.
 */
static bool parse_vector3d(PyObject *object, Vector3d &output) {
    if (!PyTuple_Check(object) || PyTuple_Size(object) != 3) {
        PyErr_SetString(PyExc_TypeError, "expected a 3-tuple of floats");
        return false;
    }
    output.x = PyFloat_AsDouble(PyTuple_GetItem(object, 0));
    output.y = PyFloat_AsDouble(PyTuple_GetItem(object, 1));
    output.z = PyFloat_AsDouble(PyTuple_GetItem(object, 2));
    return !PyErr_Occurred();
}

/**
 * @brief Parse a Python sequence of `(x, y, z)` tuples into vectors.
 *
 * @param sequence Python sequence containing point tuples.
 * @param output Parsed vector list destination.
 * @return `true` when parsing succeeds.
 */
static bool parse_vector3d_sequence(PyObject *sequence,
                                    std::vector<Vector3d> &output) {
    PyObject *fast = PySequence_Fast(sequence, "expected a sequence of 3-tuples");
    if (fast == NULL) {
        return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(fast);
    output.reserve(static_cast<std::size_t>(size));
    for (Py_ssize_t index = 0; index < size; ++index) {
        Vector3d value{};
        if (!parse_vector3d(PySequence_Fast_GET_ITEM(fast, index), value)) {
            Py_DECREF(fast);
            return false;
        }
        output.push_back(value);
    }
    Py_DECREF(fast);
    return true;
}

/**
 * @brief Parse a Python sequence of floats into a C++ vector.
 *
 * @param sequence Python sequence containing float-like values.
 * @param output Parsed float list destination.
 * @return `true` when parsing succeeds.
 */
static bool parse_double_sequence(PyObject *sequence, std::vector<double> &output) {
    PyObject *fast = PySequence_Fast(sequence, "expected a sequence of floats");
    if (fast == NULL) {
        return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(fast);
    output.reserve(static_cast<std::size_t>(size));
    for (Py_ssize_t index = 0; index < size; ++index) {
        const double value = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, index));
        if (PyErr_Occurred()) {
            Py_DECREF(fast);
            return false;
        }
        output.push_back(value);
    }
    Py_DECREF(fast);
    return true;
}

/**
 * @brief Build one Python dictionary from an octree cell.
 *
 * @param cell Source octree cell.
 * @return New Python dictionary describing the cell.
 */
static PyObject *build_octree_cell_dict(const OctreeCell &cell) {
    return Py_BuildValue(
        "{sK,sI,s((ddd)(ddd))}",
        "morton_key",
        static_cast<unsigned long long>(cell.morton_key),
        "depth",
        static_cast<unsigned int>(cell.depth),
        "bounds",
        cell.bounds.min.x,
        cell.bounds.min.y,
        cell.bounds.min.z,
        cell.bounds.max.x,
        cell.bounds.max.y,
        cell.bounds.max.z);
}

/**
 * @brief Return the name of the adaptive core status.
 *
 * The first rewrite stage exposes a very small function so tests can verify
 * that the C++ extension builds and imports successfully.
 *
 * @param self Unused Python self/module object.
 * @param args Unused Python argument tuple.
 * @return Python unicode object describing the current scaffold state.
 */
static PyObject *adaptive_status(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyUnicode_FromString("adaptive core scaffold ready");
}

/**
 * @brief Encode three integer coordinates into one Morton key.
 *
 * @param self Unused Python self/module object.
 * @param args Python argument tuple containing three unsigned integers.
 * @return Python integer holding the Morton key.
 */
static PyObject *morton_encode_3d_py(PyObject *self, PyObject *args) {
    unsigned int x = 0U;
    unsigned int y = 0U;
    unsigned int z = 0U;
    (void)self;
    if (!PyArg_ParseTuple(args, "III", &x, &y, &z)) {
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(morton_encode_3d(x, y, z));
}

/**
 * @brief Decode one Morton key into three integer coordinates.
 *
 * @param self Unused Python self/module object.
 * @param args Python argument tuple containing one unsigned integer key.
 * @return Python tuple containing the decoded coordinates.
 */
static PyObject *morton_decode_3d_py(PyObject *self, PyObject *args) {
    unsigned long long key = 0ULL;
    std::uint32_t x = 0U;
    std::uint32_t y = 0U;
    std::uint32_t z = 0U;
    (void)self;
    if (!PyArg_ParseTuple(args, "K", &key)) {
        return NULL;
    }
    morton_decode_3d(key, x, y, z);
    return Py_BuildValue("(III)", x, y, z);
}

/**
 * @brief Return whether a point lies within a bounding box.
 *
 * @param self Unused Python self/module object.
 * @param args Python tuple containing box bounds and point coordinates.
 * @return Python boolean indicating whether the point lies inside the box.
 */
static PyObject *bounding_box_contains_py(PyObject *self, PyObject *args) {
    BoundingBox box{};
    Vector3d point{};
    (void)self;
    if (!PyArg_ParseTuple(args, "(ddd)(ddd)(ddd)", &box.min.x, &box.min.y,
                          &box.min.z, &box.max.x, &box.max.y, &box.max.z,
                          &point.x, &point.y, &point.z)) {
        return NULL;
    }
    if (box.contains(point)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/**
 * @brief Return whether two bounding boxes overlap with positive volume.
 *
 * @param self Unused Python self/module object.
 * @param args Python tuple containing two bounding boxes.
 * @return Python boolean indicating whether the boxes overlap.
 */
static PyObject *bounding_box_overlaps_py(PyObject *self, PyObject *args) {
    BoundingBox left{};
    BoundingBox right{};
    (void)self;
    if (!PyArg_ParseTuple(args, "(ddd)(ddd)(ddd)(ddd)", &left.min.x,
                          &left.min.y, &left.min.z, &left.max.x, &left.max.y,
                          &left.max.z, &right.min.x, &right.min.y,
                          &right.min.z, &right.max.x, &right.max.y,
                          &right.max.z)) {
        return NULL;
    }
    if (left.overlaps(right)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/**
 * @brief Return the core particle field names used by the adaptive rewrite.
 *
 * @param self Unused Python self/module object.
 * @param args Unused Python argument tuple.
 * @return Python tuple describing the particle struct layout.
 */
static PyObject *particle_fields_py(PyObject *self, PyObject *args) {
    Particle particle{};
    (void)self;
    (void)args;
    particle.id = 0ULL;
    return Py_BuildValue(
        "(ssss)",
        "position",
        "value",
        "smoothing_length",
        "id");
}

/**
 * @brief Evaluate the Wendland C2 kernel at one radius.
 *
 * @param self Unused Python self/module object.
 * @param args Python tuple containing radius, smoothing length, and a
 *     normalization flag.
 * @return Python float containing the evaluated kernel value.
 */
static PyObject *wendland_c2_value_py(PyObject *self, PyObject *args) {
    double radius = 0.0;
    double smoothing_length = 0.0;
    int normalize = 0;
    (void)self;
    if (!PyArg_ParseTuple(args, "ddi", &radius, &smoothing_length,
                          &normalize)) {
        return NULL;
    }
    return PyFloat_FromDouble(
        evaluate_wendland_c2(radius, smoothing_length, normalize != 0));
}

/**
 * @brief Evaluate the Wendland C2 gradient for one displacement vector.
 *
 * @param self Unused Python self/module object.
 * @param args Python tuple containing displacement, smoothing length, and a
 *     normalization flag.
 * @return Python tuple containing the gradient components.
 */
static PyObject *wendland_c2_gradient_py(PyObject *self, PyObject *args) {
    Vector3d displacement{};
    double smoothing_length = 0.0;
    int normalize = 0;
    (void)self;
    if (!PyArg_ParseTuple(args, "(ddd)di", &displacement.x, &displacement.y,
                          &displacement.z, &smoothing_length, &normalize)) {
        return NULL;
    }
    const Vector3d gradient = evaluate_wendland_c2_gradient(
        displacement, smoothing_length, normalize != 0);
    return Py_BuildValue("(ddd)", gradient.x, gradient.y, gradient.z);
}

/**
 * @brief Count particles per top-level bin for one domain and resolution.
 *
 * @param self Unused Python self/module object.
 * @param args Python tuple containing particle positions, domain corners, and
 *     bin resolution.
 * @return Python tuple of flattened bin counts.
 */
static PyObject *top_level_bin_counts_py(PyObject *self, PyObject *args) {
    PyObject *positions_object = NULL;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    unsigned int resolution = 0U;
    std::vector<Vector3d> positions;
    BoundingBox domain{};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOI", &positions_object, &domain_min_object,
                          &domain_max_object, &resolution)) {
        return NULL;
    }
    if (!parse_vector3d_sequence(positions_object, positions) ||
        !parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max)) {
        return NULL;
    }

    TopLevelParticleGrid grid(domain, resolution);
    grid.insert_particles(positions);

    PyObject *counts = PyTuple_New(static_cast<Py_ssize_t>(grid.bins.size()));
    if (counts == NULL) {
        return NULL;
    }
    for (std::size_t index = 0; index < grid.bins.size(); ++index) {
        PyTuple_SET_ITEM(
            counts,
            static_cast<Py_ssize_t>(index),
            PyLong_FromSize_t(grid.bins[index].particle_indices.size()));
    }
    return counts;
}

/**
 * @brief Return candidate contributor indices for one query cell.
 *
 * @param self Unused Python self/module object.
 * @param args Python tuple containing positions, smoothing lengths, domain
 *     corners, bin resolution, and query box corners.
 * @return Python tuple of candidate particle indices.
 */
static PyObject *query_cell_contributors_py(PyObject *self, PyObject *args) {
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    PyObject *cell_min_object = NULL;
    PyObject *cell_max_object = NULL;
    unsigned int resolution = 0U;
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    BoundingBox domain{};
    BoundingBox cell{};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOIOO", &positions_object, &smoothing_object,
                          &domain_min_object, &domain_max_object, &resolution,
                          &cell_min_object, &cell_max_object)) {
        return NULL;
    }
    if (!parse_vector3d_sequence(positions_object, positions) ||
        !parse_double_sequence(smoothing_object, smoothing_lengths) ||
        !parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max) ||
        !parse_vector3d(cell_min_object, cell.min) ||
        !parse_vector3d(cell_max_object, cell.max)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths must match in size");
        return NULL;
    }

    double max_smoothing_length = 0.0;
    for (double smoothing_length : smoothing_lengths) {
        max_smoothing_length = std::max(max_smoothing_length, smoothing_length);
    }

    TopLevelParticleGrid grid(domain, resolution);
    grid.insert_particles(positions);

    std::uint32_t start_x = 0U;
    std::uint32_t start_y = 0U;
    std::uint32_t start_z = 0U;
    std::uint32_t stop_x = 0U;
    std::uint32_t stop_y = 0U;
    std::uint32_t stop_z = 0U;
    grid.contributor_bin_span(
        cell,
        max_smoothing_length,
        start_x,
        start_y,
        start_z,
        stop_x,
        stop_y,
        stop_z);

    std::vector<std::size_t> contributors;
    for (std::uint32_t ix = start_x; ix <= stop_x; ++ix) {
        for (std::uint32_t iy = start_y; iy <= stop_y; ++iy) {
            for (std::uint32_t iz = start_z; iz <= stop_z; ++iz) {
                const TopLevelBin &bin =
                    grid.bins[grid.flatten_index(ix, iy, iz)];
                for (std::size_t particle_index : bin.particle_indices) {
                    if (particle_support_overlaps_box(
                            positions[particle_index],
                            smoothing_lengths[particle_index], cell)) {
                        contributors.push_back(particle_index);
                    }
                }
            }
        }
    }

    PyObject *result = PyTuple_New(static_cast<Py_ssize_t>(contributors.size()));
    if (result == NULL) {
        return NULL;
    }
    for (std::size_t index = 0; index < contributors.size(); ++index) {
        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(index),
                         PyLong_FromSize_t(contributors[index]));
    }
    return result;
}

/**
 * @brief Return whether one cell's corners can contain the isosurface.
 */
static PyObject *cell_may_contain_isosurface_py(PyObject *self, PyObject *args) {
    PyObject *corner_values_object = NULL;
    double isovalue = 0.0;
    std::vector<double> parsed_values;
    std::array<double, 8> corner_values{};
    (void)self;
    if (!PyArg_ParseTuple(args, "Od", &corner_values_object, &isovalue)) {
        return NULL;
    }
    if (!parse_double_sequence(corner_values_object, parsed_values)) {
        return NULL;
    }
    if (parsed_values.size() != 8U) {
        PyErr_SetString(PyExc_ValueError,
                        "corner_values must contain exactly 8 samples");
        return NULL;
    }
    std::copy(parsed_values.begin(), parsed_values.end(), corner_values.begin());
    if (cell_may_contain_isosurface(corner_values, isovalue)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/**
 * @brief Return the sign mask of one cell's corner samples.
 */
static PyObject *corner_sign_mask_py(PyObject *self, PyObject *args) {
    PyObject *corner_values_object = NULL;
    double isovalue = 0.0;
    std::vector<double> parsed_values;
    std::array<double, 8> corner_values{};
    (void)self;
    if (!PyArg_ParseTuple(args, "Od", &corner_values_object, &isovalue)) {
        return NULL;
    }
    if (!parse_double_sequence(corner_values_object, parsed_values)) {
        return NULL;
    }
    if (parsed_values.size() != 8U) {
        PyErr_SetString(PyExc_ValueError,
                        "corner_values must contain exactly 8 samples");
        return NULL;
    }
    std::copy(parsed_values.begin(), parsed_values.end(), corner_values.begin());
    return PyLong_FromUnsignedLong(
        compute_corner_sign_mask(corner_values, isovalue));
}

/**
 * @brief Return a documented summary of the top-level octree cells.
 */
static PyObject *create_top_level_cells_py(PyObject *self, PyObject *args) {
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    unsigned int base_resolution = 0U;
    BoundingBox domain{};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOI", &domain_min_object, &domain_max_object,
                          &base_resolution)) {
        return NULL;
    }
    if (!parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max)) {
        return NULL;
    }
    if (base_resolution == 0U) {
        PyErr_SetString(PyExc_ValueError,
                        "base_resolution must be greater than zero");
        return NULL;
    }

    const std::vector<OctreeCell> cells =
        create_top_level_cells(domain, base_resolution);
    PyObject *result = PyTuple_New(static_cast<Py_ssize_t>(cells.size()));
    if (result == NULL) {
        return NULL;
    }
    for (std::size_t index = 0; index < cells.size(); ++index) {
        PyObject *cell_dict = build_octree_cell_dict(cells[index]);
        if (cell_dict == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(index), cell_dict);
    }
    return result;
}

/**
 * @brief Return the eight children created from one parent cell.
 */
static PyObject *create_child_cells_py(PyObject *self, PyObject *args) {
    PyObject *parent_bounds_object = NULL;
    unsigned long long parent_key = 0ULL;
    unsigned int parent_depth = 0U;
    BoundingBox parent_bounds{};
    (void)self;
    if (!PyArg_ParseTuple(args, "KOI", &parent_key, &parent_bounds_object,
                          &parent_depth)) {
        return NULL;
    }
    if (!PyTuple_Check(parent_bounds_object) || PyTuple_Size(parent_bounds_object) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "parent_bounds must be a pair of 3-tuples");
        return NULL;
    }
    if (!parse_vector3d(PyTuple_GetItem(parent_bounds_object, 0), parent_bounds.min) ||
        !parse_vector3d(PyTuple_GetItem(parent_bounds_object, 1), parent_bounds.max)) {
        return NULL;
    }

    const OctreeCell parent = {
        static_cast<std::uint64_t>(parent_key),
        static_cast<std::uint32_t>(parent_depth),
        parent_bounds,
        false,
        false,
        false,
        -1,
        -1,
        -1,
        -1,
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        0U,
    };
    const std::vector<OctreeCell> children = create_child_cells(parent);

    PyObject *result = PyTuple_New(static_cast<Py_ssize_t>(children.size()));
    if (result == NULL) {
        return NULL;
    }
    for (std::size_t index = 0; index < children.size(); ++index) {
        PyObject *child_dict = build_octree_cell_dict(children[index]);
        if (child_dict == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(index), child_dict);
    }
    return result;
}

/**
 * @brief Return contributor indices for each of a parent's eight children.
 */
static PyObject *filter_child_contributors_py(PyObject *self, PyObject *args) {
    PyObject *parent_contributors_object = NULL;
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    PyObject *parent_bounds_object = NULL;
    std::vector<double> parsed_parent_contributors;
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    BoundingBox parent_bounds{};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOO", &parent_contributors_object,
                          &positions_object, &smoothing_object,
                          &parent_bounds_object)) {
        return NULL;
    }
    if (!parse_double_sequence(parent_contributors_object, parsed_parent_contributors) ||
        !parse_vector3d_sequence(positions_object, positions) ||
        !parse_double_sequence(smoothing_object, smoothing_lengths)) {
        return NULL;
    }
    if (!PyTuple_Check(parent_bounds_object) || PyTuple_Size(parent_bounds_object) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "parent_bounds must be a pair of 3-tuples");
        return NULL;
    }
    if (!parse_vector3d(PyTuple_GetItem(parent_bounds_object, 0), parent_bounds.min) ||
        !parse_vector3d(PyTuple_GetItem(parent_bounds_object, 1), parent_bounds.max)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths must match in size");
        return NULL;
    }

    std::vector<std::size_t> parent_contributors;
    parent_contributors.reserve(parsed_parent_contributors.size());
    for (double contributor_value : parsed_parent_contributors) {
        if (contributor_value < 0.0) {
            PyErr_SetString(PyExc_ValueError,
                            "parent contributor indices must be non-negative");
            return NULL;
        }
        const std::size_t particle_index =
            static_cast<std::size_t>(contributor_value);
        if (particle_index >= positions.size()) {
            PyErr_SetString(PyExc_ValueError,
                            "parent contributor index is out of range");
            return NULL;
        }
        parent_contributors.push_back(particle_index);
    }

    const OctreeCell parent = {
        0ULL,
        0U,
        parent_bounds,
        false,
        false,
        false,
        -1,
        -1,
        -1,
        -1,
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        0U,
    };
    const std::vector<OctreeCell> children = create_child_cells(parent);
    const std::vector<std::vector<std::size_t>> child_contributors =
        filter_child_contributors(
            parent_contributors,
            positions,
            smoothing_lengths,
            children);

    PyObject *result = PyTuple_New(static_cast<Py_ssize_t>(child_contributors.size()));
    if (result == NULL) {
        return NULL;
    }
    for (std::size_t child_index = 0; child_index < child_contributors.size();
         ++child_index) {
        const std::vector<std::size_t> &contributors = child_contributors[child_index];
        PyObject *child_tuple = PyTuple_New(static_cast<Py_ssize_t>(contributors.size()));
        if (child_tuple == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        for (std::size_t index = 0; index < contributors.size(); ++index) {
            PyTuple_SET_ITEM(child_tuple, static_cast<Py_ssize_t>(index),
                             PyLong_FromSize_t(contributors[index]));
        }
        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(child_index), child_tuple);
    }
    return result;
}

/**
 * @brief Python methods exported by the adaptive extension.
 */
static PyMethodDef adaptive_methods[] = {
    {
        "adaptive_status",
        adaptive_status,
        METH_NOARGS,
        PyDoc_STR("Return the current adaptive core scaffold status."),
    },
    {
        "morton_encode_3d",
        morton_encode_3d_py,
        METH_VARARGS,
        PyDoc_STR("Encode three integer coordinates into a Morton key."),
    },
    {
        "morton_decode_3d",
        morton_decode_3d_py,
        METH_VARARGS,
        PyDoc_STR("Decode a Morton key into three integer coordinates."),
    },
    {
        "bounding_box_contains",
        bounding_box_contains_py,
        METH_VARARGS,
        PyDoc_STR("Return whether a point lies inside a bounding box."),
    },
    {
        "bounding_box_overlaps",
        bounding_box_overlaps_py,
        METH_VARARGS,
        PyDoc_STR("Return whether two bounding boxes overlap."),
    },
    {
        "particle_fields",
        particle_fields_py,
        METH_NOARGS,
        PyDoc_STR("Return the adaptive particle field names."),
    },
    {
        "wendland_c2_value",
        wendland_c2_value_py,
        METH_VARARGS,
        PyDoc_STR("Evaluate the Wendland C2 kernel value."),
    },
    {
        "wendland_c2_gradient",
        wendland_c2_gradient_py,
        METH_VARARGS,
        PyDoc_STR("Evaluate the Wendland C2 kernel gradient."),
    },
    {
        "top_level_bin_counts",
        top_level_bin_counts_py,
        METH_VARARGS,
        PyDoc_STR("Count particles per top-level bin."),
    },
    {
        "query_cell_contributors",
        query_cell_contributors_py,
        METH_VARARGS,
        PyDoc_STR("Return candidate contributor indices for a query cell."),
    },
    {
        "cell_may_contain_isosurface",
        cell_may_contain_isosurface_py,
        METH_VARARGS,
        PyDoc_STR("Return whether corner samples straddle the isosurface."),
    },
    {
        "corner_sign_mask",
        corner_sign_mask_py,
        METH_VARARGS,
        PyDoc_STR("Return the sign mask of eight corner samples."),
    },
    {
        "create_top_level_cells",
        create_top_level_cells_py,
        METH_VARARGS,
        PyDoc_STR("Create the documented top-level octree cells."),
    },
    {
        "create_child_cells",
        create_child_cells_py,
        METH_VARARGS,
        PyDoc_STR("Create the eight children of one parent cell."),
    },
    {
        "filter_child_contributors",
        filter_child_contributors_py,
        METH_VARARGS,
        PyDoc_STR("Filter parent contributors into each child cell."),
    },
    {NULL, NULL, 0, NULL},
};

/**
 * @brief Python module definition for ``meshmerizer._adaptive``.
 */
static struct PyModuleDef adaptive_module = {
    PyModuleDef_HEAD_INIT,
    "_adaptive",
    "Low-level adaptive meshing scaffold.",
    -1,
    adaptive_methods,
};

/**
 * @brief Initialize the adaptive extension module.
 *
 * @return Initialized Python module object.
 */
PyMODINIT_FUNC PyInit__adaptive(void) { return PyModule_Create(&adaptive_module); }
