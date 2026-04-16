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

#include "adaptive_cpp/bounding_box.hpp"
#include "adaptive_cpp/morton.hpp"
#include "adaptive_cpp/particle.hpp"

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
