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
 * @brief Python methods exported by the adaptive extension.
 */
static PyMethodDef adaptive_methods[] = {
    {
        "adaptive_status",
        adaptive_status,
        METH_NOARGS,
        PyDoc_STR("Return the current adaptive core scaffold status."),
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
