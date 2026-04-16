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

    TopLevelParticleGrid grid(domain, resolution);
    grid.insert_particles(positions);

    std::uint32_t start_x = 0U;
    std::uint32_t start_y = 0U;
    std::uint32_t start_z = 0U;
    std::uint32_t stop_x = 0U;
    std::uint32_t stop_y = 0U;
    std::uint32_t stop_z = 0U;
    grid.overlapping_bin_span(cell, start_x, start_y, start_z, stop_x, stop_y,
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
