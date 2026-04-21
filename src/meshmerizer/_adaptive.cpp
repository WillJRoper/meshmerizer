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

/* NumPy C API — must define NO_IMPORT_ARRAY in all translation units
 * except the one that calls import_array().  Since this is the only
 * .cpp file, we do NOT define it here (we want _import_array). */
#define PY_ARRAY_UNIQUE_SYMBOL meshmerizer_ARRAY_API
#include <numpy/arrayobject.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "adaptive_cpp/bounding_box.hpp"
#include "adaptive_cpp/hermite.hpp"
#include "adaptive_cpp/kernel_wendland_c2.hpp"
#include "adaptive_cpp/mesh.hpp"
#include "adaptive_cpp/morton.hpp"
#include "adaptive_cpp/octree_cell.hpp"
#include "adaptive_cpp/particle.hpp"
#include "adaptive_cpp/particle_grid.hpp"
#include "adaptive_cpp/progress_bar.hpp"
#include "adaptive_cpp/faces.hpp"
#include "adaptive_cpp/fof.hpp"
#include "adaptive_cpp/dc_pipeline.hpp"
#include "adaptive_cpp/adaptive_solid.hpp"
#include "adaptive_cpp/qef.hpp"

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
 * @brief Parse a Python sequence of non-negative integers into size_t.
 *
 * Each element is converted via ``PyLong_AsLong`` and validated to be
 * non-negative.  This avoids the silent truncation that occurs when
 * parsing indices through ``parse_double_sequence``.
 *
 * @param sequence Python sequence containing int-like values.
 * @param output Parsed index list destination.
 * @return ``true`` when parsing succeeds.
 */
static bool parse_index_sequence(PyObject *sequence,
                                 std::vector<std::size_t> &output) {
    PyObject *fast = PySequence_Fast(
        sequence, "expected a sequence of non-negative integers");
    if (fast == NULL) {
        return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(fast);
    output.reserve(static_cast<std::size_t>(size));
    for (Py_ssize_t index = 0; index < size; ++index) {
        const long value = PyLong_AsLong(
            PySequence_Fast_GET_ITEM(fast, index));
        if (value == -1 && PyErr_Occurred()) {
            Py_DECREF(fast);
            return false;
        }
        if (value < 0) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError,
                            "index must be non-negative");
            return false;
        }
        output.push_back(static_cast<std::size_t>(value));
    }
    Py_DECREF(fast);
    return true;
}

// ---------------------------------------------------------------------------
// Buffer-protocol parsers for NumPy arrays.
//
// These functions attempt to read positions (Nx3 float64, C-contiguous)
// and smoothing lengths (N float64, C-contiguous) directly from the
// Python buffer protocol, avoiding the per-element PyFloat_AsDouble
// overhead that dominates for 100M+ particles.  If the object does not
// support the buffer protocol (e.g., a plain Python list), the functions
// return false *without* setting a Python exception so the caller can
// fall back to the slower sequence-based parsers.
// ---------------------------------------------------------------------------

/**
 * @brief Try to parse an Nx3 C-contiguous float64 buffer into Vector3d.
 *
 * Returns true on success.  Returns false *without* setting a Python
 * exception if the object does not support the buffer protocol or has
 * the wrong shape/dtype — callers should fall back to
 * ``parse_vector3d_sequence`` in that case.  Sets a Python exception
 * only on genuine errors (e.g. buffer is float64 Nx3 but not
 * C-contiguous).
 */
static bool try_parse_positions_buffer(PyObject *object,
                                       std::vector<Vector3d> &output) {
    Py_buffer view;
    // Request C-contiguous, strided, format info.
    if (PyObject_GetBuffer(
            object, &view,
            PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
        // Object doesn't support buffer protocol — clear the error
        // and let the caller fall back to the sequence parser.
        PyErr_Clear();
        return false;
    }

    // We expect dtype=float64 (format "d") and shape (N, 3).
    bool ok = true;
    if (view.format == NULL ||
        std::strcmp(view.format, "d") != 0) {
        ok = false;
    }
    if (view.ndim != 2 || view.shape == NULL ||
        view.shape[1] != 3) {
        ok = false;
    }
    if (!ok) {
        PyBuffer_Release(&view);
        return false;
    }

    const Py_ssize_t n = view.shape[0];
    const double *data =
        static_cast<const double *>(view.buf);
    output.resize(static_cast<std::size_t>(n));

    // memcpy is valid because Vector3d is a POD struct with three
    // contiguous doubles (x, y, z) and the buffer is C-contiguous.
    static_assert(sizeof(Vector3d) == 3 * sizeof(double),
                  "Vector3d must be 3 contiguous doubles");
    std::memcpy(output.data(), data,
                static_cast<std::size_t>(n) * sizeof(Vector3d));

    PyBuffer_Release(&view);
    return true;
}

/**
 * @brief Try to parse a 1-D C-contiguous float64 buffer into doubles.
 *
 * Same fallback semantics as ``try_parse_positions_buffer``.
 */
static bool try_parse_doubles_buffer(PyObject *object,
                                     std::vector<double> &output) {
    Py_buffer view;
    if (PyObject_GetBuffer(
            object, &view,
            PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
        PyErr_Clear();
        return false;
    }

    bool ok = true;
    if (view.format == NULL ||
        std::strcmp(view.format, "d") != 0) {
        ok = false;
    }
    if (view.ndim != 1 || view.shape == NULL) {
        ok = false;
    }
    if (!ok) {
        PyBuffer_Release(&view);
        return false;
    }

    const Py_ssize_t n = view.shape[0];
    const double *data =
        static_cast<const double *>(view.buf);
    output.resize(static_cast<std::size_t>(n));
    std::memcpy(output.data(), data,
                static_cast<std::size_t>(n) * sizeof(double));

    PyBuffer_Release(&view);
    return true;
}

/**
 * @brief Parse positions from a NumPy array or Python sequence.
 *
 * Tries the fast buffer-protocol path first; falls back to the
 * element-by-element sequence parser for plain Python lists.
 */
static bool parse_positions(PyObject *object,
                            std::vector<Vector3d> &output) {
    if (try_parse_positions_buffer(object, output)) {
        return true;
    }
    return parse_vector3d_sequence(object, output);
}

/**
 * @brief Parse doubles from a NumPy array or Python sequence.
 *
 * Tries the fast buffer-protocol path first; falls back to the
 * element-by-element sequence parser for plain Python lists.
 */
static bool parse_doubles(PyObject *object,
                          std::vector<double> &output) {
    if (try_parse_doubles_buffer(object, output)) {
        return true;
    }
    return parse_double_sequence(object, output);
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
    if (!parse_positions(positions_object, positions) ||
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
    if (!parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths) ||
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
    grid.compute_bin_max_h(smoothing_lengths);

    std::uint32_t start_x = 0U;
    std::uint32_t start_y = 0U;
    std::uint32_t start_z = 0U;
    std::uint32_t stop_x = 0U;
    std::uint32_t stop_y = 0U;
    std::uint32_t stop_z = 0U;
    grid.contributor_bin_span(
        cell,
        smoothing_lengths,
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
 * @brief Create top-level cells and query contributors in one pass.
 *
 * This builds the particle grid once and reuses it for every top-level
 * cell, avoiding the O(n_cells * n_particles) cost of rebuilding the
 * grid per cell.
 *
 * Arguments: (positions, smoothing_lengths, domain_min, domain_max,
 *             base_resolution)
 *
 * Returns a tuple of cell dicts, each with an extra "contributors"
 * key holding a tuple of particle indices.
 */
static PyObject *create_top_level_cells_with_contributors_py(
        PyObject *self, PyObject *args) {
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    unsigned int base_resolution = 0U;
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    BoundingBox domain{};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOI", &positions_object,
                          &smoothing_object, &domain_min_object,
                          &domain_max_object, &base_resolution)) {
        return NULL;
    }
    if (!parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths) ||
        !parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths must match");
        return NULL;
    }
    if (base_resolution == 0U) {
        PyErr_SetString(PyExc_ValueError,
                        "base_resolution must be > 0");
        return NULL;
    }

    /* Build the particle grid once. */
    TopLevelParticleGrid grid(domain, base_resolution);
    grid.insert_particles(positions);
    grid.compute_bin_max_h(smoothing_lengths);

    /* Create top-level cells. */
    const std::vector<OctreeCell> cells =
        create_top_level_cells(domain, base_resolution);

    PyObject *result = PyTuple_New(
        static_cast<Py_ssize_t>(cells.size()));
    if (result == NULL) {
        return NULL;
    }

    ProgressBar contrib_bar("Contributor query", cells.size());
    for (std::size_t ci = 0; ci < cells.size(); ++ci) {
        /* Query contributors for this cell using the shared grid. */
        std::uint32_t sx = 0, sy = 0, sz = 0;
        std::uint32_t ex = 0, ey = 0, ez = 0;
        grid.contributor_bin_span(
            cells[ci].bounds, smoothing_lengths,
            sx, sy, sz, ex, ey, ez);

        std::vector<std::size_t> contributors;
        for (std::uint32_t ix = sx; ix <= ex; ++ix) {
            for (std::uint32_t iy = sy; iy <= ey; ++iy) {
                for (std::uint32_t iz = sz; iz <= ez; ++iz) {
                    const TopLevelBin &bin =
                        grid.bins[grid.flatten_index(ix, iy, iz)];
                    for (std::size_t pi : bin.particle_indices) {
                        if (particle_support_overlaps_box(
                                positions[pi],
                                smoothing_lengths[pi],
                                cells[ci].bounds)) {
                            contributors.push_back(pi);
                        }
                    }
                }
            }
        }

        /* Build the cell dict. */
        PyObject *cell_dict = build_octree_cell_dict(cells[ci]);
        if (cell_dict == NULL) {
            Py_DECREF(result);
            return NULL;
        }

        /* Add contributors tuple. */
        PyObject *contribs_tuple = PyTuple_New(
            static_cast<Py_ssize_t>(contributors.size()));
        if (contribs_tuple == NULL) {
            Py_DECREF(cell_dict);
            Py_DECREF(result);
            return NULL;
        }
        for (std::size_t j = 0; j < contributors.size(); ++j) {
            PyTuple_SET_ITEM(contribs_tuple,
                             static_cast<Py_ssize_t>(j),
                             PyLong_FromSize_t(contributors[j]));
        }
        if (PyDict_SetItemString(cell_dict, "contributors",
                                 contribs_tuple) < 0) {
            Py_DECREF(contribs_tuple);
            Py_DECREF(cell_dict);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(contribs_tuple);

        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(ci),
                         cell_dict);
        contrib_bar.tick();
    }
    contrib_bar.finish();
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
    std::vector<std::size_t> parent_contributors;
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    BoundingBox parent_bounds{};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOO", &parent_contributors_object,
                          &positions_object, &smoothing_object,
                          &parent_bounds_object)) {
        return NULL;
    }
    if (!parse_index_sequence(parent_contributors_object, parent_contributors) ||
        !parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths)) {
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

    // Validate contributor indices are in range.
    for (std::size_t idx : parent_contributors) {
        if (idx >= positions.size()) {
            PyErr_SetString(PyExc_ValueError,
                            "parent contributor index is out of range");
            return NULL;
        }
    }

    const OctreeCell parent = {
        0ULL,
        0U,
        parent_bounds,
        false,
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
 * @brief Build a Python dictionary from a cell with contributor indices.
 */
static PyObject *build_octree_cell_dict_with_contributors(
    const OctreeCell &cell,
    const std::vector<std::size_t> &contributors) {
    PyObject *corner_values = PyTuple_New(8);
    if (corner_values == NULL) {
        return NULL;
    }
    for (std::size_t i = 0; i < 8; ++i) {
        PyTuple_SET_ITEM(corner_values, static_cast<Py_ssize_t>(i),
                         PyFloat_FromDouble(cell.corner_values[i]));
    }

    PyObject *contrib_list = PyTuple_New(
        static_cast<Py_ssize_t>(cell.contributor_end - cell.contributor_begin));
    if (contrib_list == NULL) {
        Py_DECREF(corner_values);
        return NULL;
    }
    for (std::int64_t i = cell.contributor_begin; i < cell.contributor_end; ++i) {
        PyTuple_SET_ITEM(contrib_list, static_cast<Py_ssize_t>(i - cell.contributor_begin),
                         PyLong_FromSize_t(contributors[static_cast<std::size_t>(i)]));
    }

    return Py_BuildValue(
        "{sK,sI,s((ddd)(ddd)),sI,sI,sI,sI,sL,sB,sN,sN}",
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
        cell.bounds.max.z,
        "is_leaf",
        cell.is_leaf ? 1 : 0,
        "is_active",
        cell.is_active ? 1 : 0,
        "has_surface",
        cell.has_surface ? 1 : 0,
        "is_topo_surface",
        cell.is_topo_surface ? 1 : 0,
        "child_begin",
        static_cast<long long>(cell.child_begin),
        "corner_sign_mask",
        static_cast<unsigned int>(cell.corner_sign_mask),
        "corner_values",
        corner_values,
        "contributors",
        contrib_list);
}

/**
 * @brief Refine the octree using breadth-first refinement.
 */
static PyObject *refine_octree_py(PyObject *self, PyObject *args) {
    PyObject *initial_cells_object = NULL;
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    PyObject *domain_object = NULL;
    double isovalue = 0.0;
    unsigned int max_depth = 0U;
    unsigned int base_resolution = 0U;
    unsigned int minimum_usable_hermite_samples = 3U;
    double max_qef_rms_residual_ratio = 0.1;
    double min_normal_alignment_threshold = 0.97;
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    std::vector<OctreeCell> initial_cells;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOOdIOI|Idd", &initial_cells_object,
                          &positions_object, &smoothing_object, &isovalue,
                          &max_depth, &domain_object, &base_resolution,
                          &minimum_usable_hermite_samples,
                          &max_qef_rms_residual_ratio,
                          &min_normal_alignment_threshold)) {
        return NULL;
    }
    if (!parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths must match in size");
        return NULL;
    }

    // Parse domain bounding box from a 2-tuple of 3-tuples:
    // ((min_x, min_y, min_z), (max_x, max_y, max_z)).
    BoundingBox domain;
    if (!PyTuple_Check(domain_object) || PyTuple_Size(domain_object) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "domain must be a 2-tuple of 3-tuples");
        return NULL;
    }
    if (!parse_vector3d(PyTuple_GetItem(domain_object, 0), domain.min) ||
        !parse_vector3d(PyTuple_GetItem(domain_object, 1), domain.max)) {
        return NULL;
    }

    PyObject *cells_fast = PySequence_Fast(initial_cells_object,
                                            "expected a sequence of cell dicts");
    if (cells_fast == NULL) {
        return NULL;
    }
    const Py_ssize_t num_cells = PySequence_Fast_GET_SIZE(cells_fast);
    initial_cells.reserve(static_cast<std::size_t>(num_cells));

    for (Py_ssize_t i = 0; i < num_cells; ++i) {
        PyObject *cell_dict = PySequence_Fast_GET_ITEM(cells_fast, i);
        if (!PyDict_Check(cell_dict)) {
            Py_DECREF(cells_fast);
            PyErr_SetString(PyExc_TypeError, "each cell must be a dictionary");
            return NULL;
        }

        PyObject *morton_key_obj = PyDict_GetItemString(cell_dict, "morton_key");
        PyObject *depth_obj = PyDict_GetItemString(cell_dict, "depth");
        PyObject *bounds_obj = PyDict_GetItemString(cell_dict, "bounds");
        PyObject *contrib_begin_obj = PyDict_GetItemString(cell_dict, "contributor_begin");
        PyObject *contrib_end_obj = PyDict_GetItemString(cell_dict, "contributor_end");

        if (!morton_key_obj || !depth_obj || !bounds_obj || !contrib_begin_obj || !contrib_end_obj) {
            Py_DECREF(cells_fast);
            PyErr_SetString(PyExc_ValueError, "cell dict missing required fields");
            return NULL;
        }

        OctreeCell cell{};
        cell.morton_key = PyLong_AsUnsignedLongLong(morton_key_obj);
        cell.depth = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(depth_obj));
        cell.is_leaf = true;
        cell.is_active = false;
        cell.has_surface = false;
        cell.child_begin = -1;
        cell.representative_vertex_index = -1;
        cell.corner_sign_mask = 0U;

        if (PyTuple_Check(bounds_obj) && PyTuple_Size(bounds_obj) == 2) {
            if (!parse_vector3d(PyTuple_GetItem(bounds_obj, 0), cell.bounds.min) ||
                !parse_vector3d(PyTuple_GetItem(bounds_obj, 1), cell.bounds.max)) {
                Py_DECREF(cells_fast);
                return NULL;
            }
        } else {
            Py_DECREF(cells_fast);
            PyErr_SetString(PyExc_TypeError, "bounds must be a pair of 3-tuples");
            return NULL;
        }

        cell.contributor_begin = static_cast<std::int64_t>(PyLong_AsLong(contrib_begin_obj));
        cell.contributor_end = static_cast<std::int64_t>(PyLong_AsLong(contrib_end_obj));
        initial_cells.push_back(cell);
    }
    Py_DECREF(cells_fast);

    auto [all_cells, all_contributors] = refine_octree(
        std::move(initial_cells),
        positions,
        smoothing_lengths,
        isovalue,
        max_depth,
        domain,
        static_cast<std::uint32_t>(base_resolution),
        static_cast<std::uint32_t>(minimum_usable_hermite_samples),
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold);

    PyObject *result = PyTuple_New(2);
    if (result == NULL) {
        return NULL;
    }

    PyObject *cells_list = PyList_New(static_cast<Py_ssize_t>(all_cells.size()));
    if (cells_list == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    for (std::size_t i = 0; i < all_cells.size(); ++i) {
        PyObject *cell_dict = build_octree_cell_dict_with_contributors(
            all_cells[i], all_contributors);
        if (cell_dict == NULL) {
            Py_DECREF(cells_list);
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(cells_list, static_cast<Py_ssize_t>(i), cell_dict);
    }

    PyObject *contributors_list = PyList_New(
        static_cast<Py_ssize_t>(all_contributors.size()));
    if (contributors_list == NULL) {
        Py_DECREF(cells_list);
        Py_DECREF(result);
        return NULL;
    }
    for (std::size_t i = 0; i < all_contributors.size(); ++i) {
        PyList_SET_ITEM(contributors_list, static_cast<Py_ssize_t>(i),
                        PyLong_FromSize_t(all_contributors[i]));
    }

    PyTuple_SET_ITEM(result, 0, cells_list);
    PyTuple_SET_ITEM(result, 1, contributors_list);
    return result;
}

/**
 * @brief Return Hermite samples for one leaf cell.
 *
 * For each sign-changing edge of the cell, the function interpolates an
 * isosurface crossing point and evaluates the outward SPH gradient normal.
 *
 * @param self Unused.
 * @param args Python tuple: (bounds, corner_values, corner_sign_mask,
 *     contributor_indices, positions, smoothing_lengths, isovalue).
 * @return Python tuple of ``((px, py, pz), (nx, ny, nz))`` sample pairs.
 */
static PyObject *hermite_samples_for_cell_py(PyObject *self, PyObject *args) {
    PyObject *bounds_object = NULL;
    PyObject *corner_values_object = NULL;
    unsigned int corner_sign_mask_uint = 0U;
    PyObject *contributors_object = NULL;
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    double isovalue = 0.0;
    BoundingBox bounds{};
    std::vector<double> parsed_corner_values;
    std::array<double, 8> corner_values{};
    std::vector<std::size_t> contributor_indices;
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOIOOOd",
                          &bounds_object,
                          &corner_values_object,
                          &corner_sign_mask_uint,
                          &contributors_object,
                          &positions_object,
                          &smoothing_object,
                          &isovalue)) {
        return NULL;
    }

    if (!PyTuple_Check(bounds_object) || PyTuple_Size(bounds_object) != 2) {
        PyErr_SetString(PyExc_TypeError, "bounds must be a pair of 3-tuples");
        return NULL;
    }
    if (!parse_vector3d(PyTuple_GetItem(bounds_object, 0), bounds.min) ||
        !parse_vector3d(PyTuple_GetItem(bounds_object, 1), bounds.max)) {
        return NULL;
    }

    if (!parse_double_sequence(corner_values_object, parsed_corner_values) ||
        !parse_index_sequence(contributors_object, contributor_indices) ||
        !parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths)) {
        return NULL;
    }
    if (parsed_corner_values.size() != 8U) {
        PyErr_SetString(PyExc_ValueError,
                        "corner_values must contain exactly 8 samples");
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing_lengths must match in size");
        return NULL;
    }

    std::copy(parsed_corner_values.begin(), parsed_corner_values.end(),
              corner_values.begin());

    // Validate contributor indices are in range.
    for (std::size_t idx : contributor_indices) {
        if (idx >= positions.size()) {
            PyErr_SetString(PyExc_ValueError,
                            "contributor index out of range");
            return NULL;
        }
    }

    const std::uint8_t corner_sign_mask =
        static_cast<std::uint8_t>(corner_sign_mask_uint);

    const std::vector<HermiteSample> samples = compute_cell_hermite_samples(
        bounds, corner_values, corner_sign_mask, contributor_indices,
        positions, smoothing_lengths, isovalue);

    PyObject *result = PyTuple_New(static_cast<Py_ssize_t>(samples.size()));
    if (result == NULL) {
        return NULL;
    }
    for (std::size_t i = 0; i < samples.size(); ++i) {
        PyObject *sample = Py_BuildValue(
            "((ddd)(ddd))",
            samples[i].position.x,
            samples[i].position.y,
            samples[i].position.z,
            samples[i].normal.x,
            samples[i].normal.y,
            samples[i].normal.z);
        if (sample == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(i), sample);
    }
    return result;
}

/**
 * @brief Build top-level cells and refine them in one resumable call.
 *
 * This binding mirrors the first half of ``run_octree_pipeline`` but returns
 * the refined cell dictionaries and flat contributor array so Python callers
 * can continue the pipeline from an explicit tree state.
 */
static PyObject *build_refined_tree_py(PyObject * /* self */, PyObject *args) {
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    unsigned int base_resolution = 0U;
    double isovalue = 0.0;
    unsigned int max_depth = 0U;
    unsigned int minimum_usable_hermite_samples = 3U;
    double max_qef_rms_residual_ratio = 0.1;
    double min_normal_alignment_threshold = 0.97;

    if (!PyArg_ParseTuple(args, "OOOOIdI|Idd",
                          &positions_object,
                          &smoothing_object,
                          &domain_min_object,
                          &domain_max_object,
                          &base_resolution,
                          &isovalue,
                          &max_depth,
                          &minimum_usable_hermite_samples,
                          &max_qef_rms_residual_ratio,
                          &min_normal_alignment_threshold)) {
        return NULL;
    }

    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    if (!parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths must match in size");
        return NULL;
    }

    BoundingBox domain{};
    if (!parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max)) {
        return NULL;
    }

    TopLevelParticleGrid grid(domain, base_resolution);
    grid.insert_particles(positions);
    grid.compute_bin_max_h(smoothing_lengths);

    std::vector<OctreeCell> top_cells =
        create_top_level_cells(domain, base_resolution);
    std::vector<OctreeCell> initial_cells;
    initial_cells.reserve(top_cells.size());
    std::vector<std::size_t> initial_contributors;

    ProgressBar pipeline_contrib_bar(
        "Contributor query", top_cells.size());
    for (std::size_t ci = 0; ci < top_cells.size(); ++ci) {
        OctreeCell cell = top_cells[ci];

        std::uint32_t sx = 0, sy = 0, sz = 0;
        std::uint32_t ex = 0, ey = 0, ez = 0;
        grid.contributor_bin_span(
            cell.bounds, smoothing_lengths,
            sx, sy, sz, ex, ey, ez);

        const std::int64_t begin = static_cast<std::int64_t>(
            initial_contributors.size());

        for (std::uint32_t ix = sx; ix <= ex; ++ix) {
            for (std::uint32_t iy = sy; iy <= ey; ++iy) {
                for (std::uint32_t iz = sz; iz <= ez; ++iz) {
                    const TopLevelBin &bin =
                        grid.bins[grid.flatten_index(ix, iy, iz)];
                    for (std::size_t pi : bin.particle_indices) {
                        if (particle_support_overlaps_box(
                                positions[pi],
                                smoothing_lengths[pi],
                                cell.bounds)) {
                            initial_contributors.push_back(pi);
                        }
                    }
                }
            }
        }

        const std::int64_t end = static_cast<std::int64_t>(
            initial_contributors.size());
        cell.contributor_begin = begin;
        cell.contributor_end = end;
        initial_cells.push_back(cell);
        pipeline_contrib_bar.tick();
    }
    pipeline_contrib_bar.finish();

    auto [all_cells, all_contributors] = refine_octree(
        std::move(initial_cells),
        std::move(initial_contributors),
        positions,
        smoothing_lengths,
        isovalue,
        max_depth,
        domain,
        static_cast<std::uint32_t>(base_resolution),
        static_cast<std::uint32_t>(minimum_usable_hermite_samples),
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold);

    PyObject *cells_list = PyList_New(static_cast<Py_ssize_t>(all_cells.size()));
    if (cells_list == NULL) {
        return NULL;
    }
    for (std::size_t i = 0; i < all_cells.size(); ++i) {
        PyObject *cell_dict = build_octree_cell_dict_with_contributors(
            all_cells[i], all_contributors);
        if (cell_dict == NULL) {
            Py_DECREF(cells_list);
            return NULL;
        }
        PyList_SET_ITEM(cells_list, static_cast<Py_ssize_t>(i), cell_dict);
    }

    npy_intp contrib_dims[1] = {
        static_cast<npy_intp>(all_contributors.size())};
    PyObject *contributors_array =
        PyArray_SimpleNew(1, contrib_dims, NPY_INT64);
    if (contributors_array == NULL) {
        Py_DECREF(cells_list);
        return NULL;
    }
    auto *contrib_data = static_cast<std::int64_t *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(contributors_array)));
    for (std::size_t i = 0; i < all_contributors.size(); ++i) {
        contrib_data[i] = static_cast<std::int64_t>(all_contributors[i]);
    }

    return Py_BuildValue("(NN)", cells_list, contributors_array);
}

/**
 * @brief Solve the QEF for one leaf cell and return its representative vertex.
 *
 * @param self Unused.
 * @param args Python tuple: (samples, bounds).  ``samples`` is a sequence of
 *     ``((px, py, pz), (nx, ny, nz))`` pairs.  ``bounds`` is a
 *     ``((min_x, min_y, min_z), (max_x, max_y, max_z))`` pair.
 * @return Python tuple ``((px, py, pz), (nx, ny, nz))`` for the vertex.
 */
static PyObject *solve_qef_for_leaf_py(PyObject *self, PyObject *args) {
    PyObject *samples_object = NULL;
    PyObject *bounds_object = NULL;
    BoundingBox bounds{};
    (void)self;

    if (!PyArg_ParseTuple(args, "OO", &samples_object, &bounds_object)) {
        return NULL;
    }

    if (!PyTuple_Check(bounds_object) || PyTuple_Size(bounds_object) != 2) {
        PyErr_SetString(PyExc_TypeError, "bounds must be a pair of 3-tuples");
        return NULL;
    }
    if (!parse_vector3d(PyTuple_GetItem(bounds_object, 0), bounds.min) ||
        !parse_vector3d(PyTuple_GetItem(bounds_object, 1), bounds.max)) {
        return NULL;
    }

    PyObject *samples_fast = PySequence_Fast(samples_object,
                                             "expected a sequence of sample pairs");
    if (samples_fast == NULL) {
        return NULL;
    }

    const Py_ssize_t num_samples = PySequence_Fast_GET_SIZE(samples_fast);
    std::vector<HermiteSample> samples;
    samples.reserve(static_cast<std::size_t>(num_samples));

    for (Py_ssize_t i = 0; i < num_samples; ++i) {
        PyObject *pair = PySequence_Fast_GET_ITEM(samples_fast, i);
        if (!PyTuple_Check(pair) || PyTuple_Size(pair) != 2) {
            Py_DECREF(samples_fast);
            PyErr_SetString(PyExc_TypeError,
                            "each sample must be a pair of 3-tuples");
            return NULL;
        }
        HermiteSample sample{};
        if (!parse_vector3d(PyTuple_GetItem(pair, 0), sample.position) ||
            !parse_vector3d(PyTuple_GetItem(pair, 1), sample.normal)) {
            Py_DECREF(samples_fast);
            return NULL;
        }
        samples.push_back(sample);
    }
    Py_DECREF(samples_fast);

    const MeshVertex vertex = solve_qef_for_leaf(samples, bounds);

    return Py_BuildValue(
        "((ddd)(ddd))",
        vertex.position.x,
        vertex.position.y,
        vertex.position.z,
        vertex.normal.x,
        vertex.normal.y,
        vertex.normal.z);
}

/**
 * @brief Build a Python tuple of (positions, normals, triangles) as NumPy
 *        arrays from the C++ mesh output.
 *
 * Returns (positions_Nx3_float64, normals_Nx3_float64, triangles_Mx3_int64).
 * On failure sets a Python exception and returns NULL.
 */

/* Forward declaration — used by solve_vertices_py before definition. */
static PyObject *build_vertices_numpy_result(
    const std::vector<MeshVertex> &vertices);

static PyObject *build_mesh_numpy_result(
    const std::vector<MeshVertex> &vertices,
    const std::vector<MeshTriangle> &triangles) {
    // -- Vertex positions: Nx3 float64 --
    const npy_intp n_verts = static_cast<npy_intp>(vertices.size());
    npy_intp pos_dims[2] = {n_verts, 3};
    PyObject *pos_arr = PyArray_SimpleNew(2, pos_dims, NPY_DOUBLE);
    if (pos_arr == NULL) return NULL;
    {
        double *data = static_cast<double *>(
            PyArray_DATA(reinterpret_cast<PyArrayObject *>(pos_arr)));
        for (npy_intp i = 0; i < n_verts; ++i) {
            data[i * 3 + 0] = vertices[static_cast<std::size_t>(i)]
                                   .position.x;
            data[i * 3 + 1] = vertices[static_cast<std::size_t>(i)]
                                   .position.y;
            data[i * 3 + 2] = vertices[static_cast<std::size_t>(i)]
                                   .position.z;
        }
    }

    // -- Vertex normals: Nx3 float64 --
    PyObject *norm_arr = PyArray_SimpleNew(2, pos_dims, NPY_DOUBLE);
    if (norm_arr == NULL) {
        Py_DECREF(pos_arr);
        return NULL;
    }
    {
        double *data = static_cast<double *>(
            PyArray_DATA(reinterpret_cast<PyArrayObject *>(norm_arr)));
        for (npy_intp i = 0; i < n_verts; ++i) {
            data[i * 3 + 0] = vertices[static_cast<std::size_t>(i)]
                                   .normal.x;
            data[i * 3 + 1] = vertices[static_cast<std::size_t>(i)]
                                   .normal.y;
            data[i * 3 + 2] = vertices[static_cast<std::size_t>(i)]
                                   .normal.z;
        }
    }

    // -- Triangle indices: Mx3 int64 --
    const npy_intp n_tris = static_cast<npy_intp>(triangles.size());
    npy_intp tri_dims[2] = {n_tris, 3};
    PyObject *tri_arr = PyArray_SimpleNew(2, tri_dims, NPY_INT64);
    if (tri_arr == NULL) {
        Py_DECREF(pos_arr);
        Py_DECREF(norm_arr);
        return NULL;
    }
    {
        std::int64_t *data = static_cast<std::int64_t *>(
            PyArray_DATA(reinterpret_cast<PyArrayObject *>(tri_arr)));
        for (npy_intp i = 0; i < n_tris; ++i) {
            data[i * 3 + 0] = static_cast<std::int64_t>(
                triangles[static_cast<std::size_t>(i)]
                    .vertex_indices[0]);
            data[i * 3 + 1] = static_cast<std::int64_t>(
                triangles[static_cast<std::size_t>(i)]
                    .vertex_indices[1]);
            data[i * 3 + 2] = static_cast<std::int64_t>(
                triangles[static_cast<std::size_t>(i)]
                    .vertex_indices[2]);
        }
    }

    // -- Build the result tuple manually so we can clean up on error.
    // Py_BuildValue("NNN") steals references, which means if it fails
    // partway through, already-stolen refs are lost.  Using explicit
    // PyTuple_New + PyTuple_SET_ITEM avoids this: SET_ITEM steals the
    // ref only on success, and on any allocation failure we can still
    // Py_DECREF all three arrays. --
    PyObject *result = PyTuple_New(3);
    if (result == NULL) {
        Py_DECREF(pos_arr);
        Py_DECREF(norm_arr);
        Py_DECREF(tri_arr);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, pos_arr);   // steals ref
    PyTuple_SET_ITEM(result, 1, norm_arr);  // steals ref
    PyTuple_SET_ITEM(result, 2, tri_arr);   // steals ref
    return result;
}

/**
 * @brief Run the full mesh generation pipeline on a refined octree.
 *
 * Takes the output of ``refine_octree`` (cells and contributors) along
 * with particle data, and produces a triangle mesh via dual contouring.
 *
 * @param self Unused.
 * @param args Python tuple: (cells, contributors, positions,
 *     smoothing_lengths, isovalue, domain_min, domain_max,
 *     max_depth, base_resolution).
 * @return Python tuple: (positions, normals, triangles) where positions
 *     is an Nx3 float64 NumPy array, normals is an Nx3 float64 NumPy
 *     array, and triangles is an Mx3 int64 NumPy array of vertex
 *     index triples.
 */
static PyObject *generate_mesh_py(PyObject *self, PyObject *args) {
    PyObject *cells_object = NULL;
    PyObject *contributors_object = NULL;
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    double isovalue = 0.0;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    unsigned int max_depth = 0U;
    unsigned int base_resolution = 0U;
    BoundingBox domain{};
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOOOdOOII",
                          &cells_object,
                          &contributors_object,
                          &positions_object,
                          &smoothing_object,
                          &isovalue,
                          &domain_min_object,
                          &domain_max_object,
                          &max_depth,
                          &base_resolution)) {
        return NULL;
    }

    if (!parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths must match");
        return NULL;
    }
    if (!parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max)) {
        return NULL;
    }

    // Parse the contributor array.
    PyObject *contrib_fast = PySequence_Fast(
        contributors_object, "expected a sequence of ints");
    if (contrib_fast == NULL) {
        return NULL;
    }
    const Py_ssize_t num_contrib =
        PySequence_Fast_GET_SIZE(contrib_fast);
    std::vector<std::size_t> all_contributors;
    all_contributors.reserve(static_cast<std::size_t>(num_contrib));
    for (Py_ssize_t i = 0; i < num_contrib; ++i) {
        long val = PyLong_AsLong(
            PySequence_Fast_GET_ITEM(contrib_fast, i));
        if (val == -1 && PyErr_Occurred()) {
            Py_DECREF(contrib_fast);
            return NULL;
        }
        if (val < 0) {
            Py_DECREF(contrib_fast);
            PyErr_SetString(PyExc_ValueError,
                            "contributor indices must be non-negative");
            return NULL;
        }
        all_contributors.push_back(static_cast<std::size_t>(val));
    }
    Py_DECREF(contrib_fast);

    // Parse the cells array.
    PyObject *cells_fast = PySequence_Fast(
        cells_object, "expected a sequence of cell dicts");
    if (cells_fast == NULL) {
        return NULL;
    }
    const Py_ssize_t num_cells =
        PySequence_Fast_GET_SIZE(cells_fast);
    std::vector<OctreeCell> all_cells;
    all_cells.reserve(static_cast<std::size_t>(num_cells));

    for (Py_ssize_t i = 0; i < num_cells; ++i) {
        PyObject *d = PySequence_Fast_GET_ITEM(cells_fast, i);
        if (!PyDict_Check(d)) {
            Py_DECREF(cells_fast);
            PyErr_SetString(PyExc_TypeError,
                            "each cell must be a dictionary");
            return NULL;
        }

        OctreeCell cell{};
        cell.morton_key = PyLong_AsUnsignedLongLong(
            PyDict_GetItemString(d, "morton_key"));
        cell.depth = static_cast<std::uint32_t>(
            PyLong_AsUnsignedLong(
                PyDict_GetItemString(d, "depth")));

        PyObject *bounds_obj = PyDict_GetItemString(d, "bounds");
        if (bounds_obj && PyTuple_Check(bounds_obj) &&
            PyTuple_Size(bounds_obj) == 2) {
            parse_vector3d(
                PyTuple_GetItem(bounds_obj, 0), cell.bounds.min);
            parse_vector3d(
                PyTuple_GetItem(bounds_obj, 1), cell.bounds.max);
        }

        PyObject *is_leaf_obj =
            PyDict_GetItemString(d, "is_leaf");
        cell.is_leaf = is_leaf_obj
            ? (PyLong_AsLong(is_leaf_obj) != 0)
            : true;

        PyObject *has_surface_obj =
            PyDict_GetItemString(d, "has_surface");
        cell.has_surface = has_surface_obj
            ? (PyLong_AsLong(has_surface_obj) != 0)
            : false;

        PyObject *is_active_obj =
            PyDict_GetItemString(d, "is_active");
        cell.is_active = is_active_obj
            ? (PyLong_AsLong(is_active_obj) != 0)
            : false;

        PyObject *child_begin_obj =
            PyDict_GetItemString(d, "child_begin");
        cell.child_begin = child_begin_obj
            ? static_cast<std::int64_t>(
                  PyLong_AsLong(child_begin_obj))
            : -1;

        cell.representative_vertex_index = -1;

        // Parse corner_sign_mask.
        PyObject *csm_obj =
            PyDict_GetItemString(d, "corner_sign_mask");
        cell.corner_sign_mask = csm_obj
            ? static_cast<std::uint8_t>(
                  PyLong_AsUnsignedLong(csm_obj))
            : 0U;

        // Parse corner_values.
        PyObject *cv_obj =
            PyDict_GetItemString(d, "corner_values");
        if (cv_obj) {
            std::vector<double> cv;
            if (parse_doubles(cv_obj, cv) &&
                cv.size() == 8U) {
                std::copy(cv.begin(), cv.end(),
                          cell.corner_values.begin());
            }
        }

        // Parse contributor range.
        PyObject *cb_obj =
            PyDict_GetItemString(d, "contributor_begin");
        PyObject *ce_obj =
            PyDict_GetItemString(d, "contributor_end");
        if (cb_obj && ce_obj) {
            cell.contributor_begin =
                static_cast<std::int64_t>(
                    PyLong_AsLong(cb_obj));
            cell.contributor_end =
                static_cast<std::int64_t>(
                    PyLong_AsLong(ce_obj));
        } else {
            // Fall back to contributors list if present.
            PyObject *contribs_obj =
                PyDict_GetItemString(d, "contributors");
            if (contribs_obj) {
                PyObject *cfast = PySequence_Fast(
                    contribs_obj, "contributors");
                if (cfast) {
                    Py_ssize_t nc =
                        PySequence_Fast_GET_SIZE(cfast);
                    cell.contributor_begin =
                        static_cast<std::int64_t>(
                            all_contributors.size());
                    for (Py_ssize_t ci = 0; ci < nc; ++ci) {
                        long v = PyLong_AsLong(
                            PySequence_Fast_GET_ITEM(
                                cfast, ci));
                        all_contributors.push_back(
                            static_cast<std::size_t>(v));
                    }
                    cell.contributor_end =
                        static_cast<std::int64_t>(
                            all_contributors.size());
                    Py_DECREF(cfast);
                }
            } else {
                cell.contributor_begin = -1;
                cell.contributor_end = -1;
            }
        }

        if (PyErr_Occurred()) {
            Py_DECREF(cells_fast);
            return NULL;
        }
        all_cells.push_back(cell);
    }
    Py_DECREF(cells_fast);

    // Run the mesh generation pipeline.
    auto [vertices, triangles] = generate_mesh(
        all_cells, all_contributors, positions, smoothing_lengths,
        isovalue, domain, max_depth, base_resolution);

    // Return (positions_Nx3, normals_Nx3, triangles_Mx3) as NumPy arrays.
    return build_mesh_numpy_result(vertices, triangles);
}

/**
 * @brief Solve QEF vertices for all active leaf cells (vertex-only,
 *        no face generation).
 *
 * It accepts the same arguments but returns only ``(positions, normals)``
 * without triangle faces. This is mainly used for diagnostics,
 * visualization, and tests that inspect the vertex solve separately from the
 * final mesh extraction path.
 *
 * @param args  Python tuple:
 *   (cells, contributors, positions, smoothing_lengths, isovalue,
 *    domain_min, domain_max, max_depth, base_resolution)
 * @return  ``(positions_Nx3, normals_Nx3)`` NumPy arrays.
 */
static PyObject *solve_vertices_py(PyObject *self, PyObject *args) {
    PyObject *cells_object = NULL;
    PyObject *contributors_object = NULL;
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    double isovalue = 0.0;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    unsigned int max_depth = 0U;
    unsigned int base_resolution = 0U;
    BoundingBox domain{};
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOOOdOOII",
                          &cells_object,
                          &contributors_object,
                          &positions_object,
                          &smoothing_object,
                          &isovalue,
                          &domain_min_object,
                          &domain_max_object,
                          &max_depth,
                          &base_resolution)) {
        return NULL;
    }

    if (!parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths must match");
        return NULL;
    }
    if (!parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max)) {
        return NULL;
    }

    // Parse the contributor array.
    PyObject *contrib_fast = PySequence_Fast(
        contributors_object, "expected a sequence of ints");
    if (contrib_fast == NULL) {
        return NULL;
    }
    const Py_ssize_t num_contrib =
        PySequence_Fast_GET_SIZE(contrib_fast);
    std::vector<std::size_t> all_contributors;
    all_contributors.reserve(static_cast<std::size_t>(num_contrib));
    for (Py_ssize_t i = 0; i < num_contrib; ++i) {
        long val = PyLong_AsLong(
            PySequence_Fast_GET_ITEM(contrib_fast, i));
        if (val == -1 && PyErr_Occurred()) {
            Py_DECREF(contrib_fast);
            return NULL;
        }
        if (val < 0) {
            Py_DECREF(contrib_fast);
            PyErr_SetString(PyExc_ValueError,
                            "contributor indices must be non-negative");
            return NULL;
        }
        all_contributors.push_back(static_cast<std::size_t>(val));
    }
    Py_DECREF(contrib_fast);

    // Parse the cells array.
    PyObject *cells_fast = PySequence_Fast(
        cells_object, "expected a sequence of cell dicts");
    if (cells_fast == NULL) {
        return NULL;
    }
    const Py_ssize_t num_cells =
        PySequence_Fast_GET_SIZE(cells_fast);
    std::vector<OctreeCell> all_cells;
    all_cells.reserve(static_cast<std::size_t>(num_cells));

    for (Py_ssize_t i = 0; i < num_cells; ++i) {
        PyObject *d = PySequence_Fast_GET_ITEM(cells_fast, i);
        if (!PyDict_Check(d)) {
            Py_DECREF(cells_fast);
            PyErr_SetString(PyExc_TypeError,
                            "each cell must be a dictionary");
            return NULL;
        }

        OctreeCell cell{};
        cell.morton_key = PyLong_AsUnsignedLongLong(
            PyDict_GetItemString(d, "morton_key"));
        cell.depth = static_cast<std::uint32_t>(
            PyLong_AsUnsignedLong(
                PyDict_GetItemString(d, "depth")));

        PyObject *bounds_obj = PyDict_GetItemString(d, "bounds");
        if (bounds_obj && PyTuple_Check(bounds_obj) &&
            PyTuple_Size(bounds_obj) == 2) {
            parse_vector3d(
                PyTuple_GetItem(bounds_obj, 0), cell.bounds.min);
            parse_vector3d(
                PyTuple_GetItem(bounds_obj, 1), cell.bounds.max);
        }

        PyObject *is_leaf_obj =
            PyDict_GetItemString(d, "is_leaf");
        cell.is_leaf = is_leaf_obj
            ? (PyLong_AsLong(is_leaf_obj) != 0)
            : true;

        PyObject *has_surface_obj =
            PyDict_GetItemString(d, "has_surface");
        cell.has_surface = has_surface_obj
            ? (PyLong_AsLong(has_surface_obj) != 0)
            : false;

        PyObject *is_active_obj =
            PyDict_GetItemString(d, "is_active");
        cell.is_active = is_active_obj
            ? (PyLong_AsLong(is_active_obj) != 0)
            : false;

        PyObject *child_begin_obj =
            PyDict_GetItemString(d, "child_begin");
        cell.child_begin = child_begin_obj
            ? static_cast<std::int64_t>(
                  PyLong_AsLong(child_begin_obj))
            : -1;

        cell.representative_vertex_index = -1;

        // Parse corner_sign_mask.
        PyObject *csm_obj =
            PyDict_GetItemString(d, "corner_sign_mask");
        cell.corner_sign_mask = csm_obj
            ? static_cast<std::uint8_t>(
                  PyLong_AsUnsignedLong(csm_obj))
            : 0U;

        // Parse corner_values.
        PyObject *cv_obj =
            PyDict_GetItemString(d, "corner_values");
        if (cv_obj) {
            std::vector<double> cv;
            if (parse_doubles(cv_obj, cv) &&
                cv.size() == 8U) {
                std::copy(cv.begin(), cv.end(),
                          cell.corner_values.begin());
            }
        }

        // Parse contributor range.
        PyObject *cb_obj =
            PyDict_GetItemString(d, "contributor_begin");
        PyObject *ce_obj =
            PyDict_GetItemString(d, "contributor_end");
        if (cb_obj && ce_obj) {
            cell.contributor_begin =
                static_cast<std::int64_t>(
                    PyLong_AsLong(cb_obj));
            cell.contributor_end =
                static_cast<std::int64_t>(
                    PyLong_AsLong(ce_obj));
        } else {
            // Fall back to contributors list if present.
            PyObject *contribs_obj =
                PyDict_GetItemString(d, "contributors");
            if (contribs_obj) {
                PyObject *cfast = PySequence_Fast(
                    contribs_obj, "contributors");
                if (cfast) {
                    Py_ssize_t nc =
                        PySequence_Fast_GET_SIZE(cfast);
                    cell.contributor_begin =
                        static_cast<std::int64_t>(
                            all_contributors.size());
                    for (Py_ssize_t ci = 0; ci < nc; ++ci) {
                        long v = PyLong_AsLong(
                            PySequence_Fast_GET_ITEM(
                                cfast, ci));
                        all_contributors.push_back(
                            static_cast<std::size_t>(v));
                    }
                    cell.contributor_end =
                        static_cast<std::int64_t>(
                            all_contributors.size());
                    Py_DECREF(cfast);
                }
            } else {
                cell.contributor_begin = -1;
                cell.contributor_end = -1;
            }
        }

        if (PyErr_Occurred()) {
            Py_DECREF(cells_fast);
            return NULL;
        }
        all_cells.push_back(cell);
    }
    Py_DECREF(cells_fast);

    // Solve QEF vertices only (no face generation).
    std::vector<MeshVertex> vertices = solve_all_leaf_vertices(
        all_cells, all_contributors, positions, smoothing_lengths,
        isovalue);

    // Return (positions_Nx3, normals_Nx3) as NumPy arrays.
    return build_vertices_numpy_result(vertices);
}

/**
 * @brief Build a (positions_Nx3, normals_Nx3) Python tuple from
 *        a vector of MeshVertex.
 *
 * This is the vertex-only counterpart to ``build_mesh_numpy_result``,
 * used by the octree pipeline which returns QEF vertices without
 * triangle faces.
 */
static PyObject *build_vertices_numpy_result(
    const std::vector<MeshVertex> &vertices) {
    const npy_intp n_verts = static_cast<npy_intp>(vertices.size());
    npy_intp dims[2] = {n_verts, 3};

    // -- Vertex positions: Nx3 float64 --
    PyObject *pos_arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (pos_arr == NULL) return NULL;
    {
        double *data = static_cast<double *>(
            PyArray_DATA(
                reinterpret_cast<PyArrayObject *>(pos_arr)));
        for (npy_intp i = 0; i < n_verts; ++i) {
            data[i * 3 + 0] = vertices[
                static_cast<std::size_t>(i)].position.x;
            data[i * 3 + 1] = vertices[
                static_cast<std::size_t>(i)].position.y;
            data[i * 3 + 2] = vertices[
                static_cast<std::size_t>(i)].position.z;
        }
    }

    // -- Vertex normals: Nx3 float64 --
    PyObject *norm_arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (norm_arr == NULL) {
        Py_DECREF(pos_arr);
        return NULL;
    }
    {
        double *data = static_cast<double *>(
            PyArray_DATA(
                reinterpret_cast<PyArrayObject *>(norm_arr)));
        for (npy_intp i = 0; i < n_verts; ++i) {
            data[i * 3 + 0] = vertices[
                static_cast<std::size_t>(i)].normal.x;
            data[i * 3 + 1] = vertices[
                static_cast<std::size_t>(i)].normal.y;
            data[i * 3 + 2] = vertices[
                static_cast<std::size_t>(i)].normal.z;
        }
    }

    // Build result tuple.
    PyObject *result = PyTuple_New(2);
    if (result == NULL) {
        Py_DECREF(pos_arr);
        Py_DECREF(norm_arr);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, pos_arr);   // steals ref
    PyTuple_SET_ITEM(result, 1, norm_arr);  // steals ref
    return result;
}

/**
 * @brief Run the octree pipeline in C++ and return QEF vertices.
 *
 * This builds the adaptive octree and solves QEF vertices for every
 * active leaf cell, but does NOT generate triangle faces.  Face
 * generation is handled later by the reconstruction pipeline.
 *
 * Accepts:
 *   positions         – Nx3 float64 array (or list of 3-tuples)
 *   smoothing_lengths – N float64 array (or list of floats)
 *   domain_min        – (x, y, z) tuple
 *   domain_max        – (x, y, z) tuple
 *   base_resolution   – unsigned int
 *   isovalue          – double
 *   max_depth         – unsigned int
 *
 * Returns (positions_Nx3, normals_Nx3) as NumPy float64 arrays
 * where N is the number of active leaf cells with surface crossings.
 *
 * This combines create_top_level_cells + contributor query +
 * refine_octree + solve_all_leaf_vertices into one C++ call,
 * eliminating the massive serialization overhead of passing octree
 * cells and contributor arrays through Python dictionaries.
 */
static PyObject *run_octree_pipeline_py(
        PyObject *self, PyObject *args) {
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    unsigned int base_resolution = 0U;
    double isovalue = 0.0;
    unsigned int max_depth = 0U;
    unsigned int minimum_usable_hermite_samples = 3U;
    double max_qef_rms_residual_ratio = 0.1;
    double min_normal_alignment_threshold = 0.97;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOOOIdI|Idd",
                          &positions_object,
                          &smoothing_object,
                          &domain_min_object,
                          &domain_max_object,
                          &base_resolution,
                          &isovalue,
                          &max_depth,
                          &minimum_usable_hermite_samples,
                          &max_qef_rms_residual_ratio,
                          &min_normal_alignment_threshold)) {
        return NULL;
    }

    // Parse particle data (prefers NumPy buffer protocol).
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    BoundingBox domain{};

    if (!parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths "
                        "must match in size");
        return NULL;
    }
    if (!parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max)) {
        return NULL;
    }
    if (base_resolution == 0U) {
        PyErr_SetString(PyExc_ValueError,
                        "base_resolution must be > 0");
        return NULL;
    }

    // Release the GIL for the entire C++ pipeline since we no
    // longer need any Python objects after parsing.
    std::vector<MeshVertex> vertices;

    Py_BEGIN_ALLOW_THREADS

    // -- Step 1: Create top-level cells + query contributors --
    TopLevelParticleGrid grid(domain, base_resolution);
    grid.insert_particles(positions);
    grid.compute_bin_max_h(smoothing_lengths);

    std::vector<OctreeCell> top_cells =
        create_top_level_cells(domain, base_resolution);

    // Build initial cells with contributor ranges stored in a
    // flat vector that refine_octree can consume directly.
    std::vector<OctreeCell> initial_cells;
    initial_cells.reserve(top_cells.size());
    std::vector<std::size_t> initial_contributors;

    ProgressBar pipeline_contrib_bar(
        "Contributor query", top_cells.size());
    for (std::size_t ci = 0; ci < top_cells.size(); ++ci) {
        OctreeCell cell = top_cells[ci];

        std::uint32_t sx = 0, sy = 0, sz = 0;
        std::uint32_t ex = 0, ey = 0, ez = 0;
        grid.contributor_bin_span(
            cell.bounds, smoothing_lengths,
            sx, sy, sz, ex, ey, ez);

        const std::int64_t begin =
            static_cast<std::int64_t>(
                initial_contributors.size());

        for (std::uint32_t ix = sx; ix <= ex; ++ix) {
            for (std::uint32_t iy = sy; iy <= ey; ++iy) {
                for (std::uint32_t iz = sz;
                     iz <= ez; ++iz) {
                    const TopLevelBin &bin =
                        grid.bins[grid.flatten_index(
                            ix, iy, iz)];
                    for (std::size_t pi :
                         bin.particle_indices) {
                        if (particle_support_overlaps_box(
                                positions[pi],
                                smoothing_lengths[pi],
                                cell.bounds)) {
                            initial_contributors.push_back(
                                pi);
                        }
                    }
                }
            }
        }

        const std::int64_t end =
            static_cast<std::int64_t>(
                initial_contributors.size());
        cell.contributor_begin = begin;
        cell.contributor_end = end;
        initial_cells.push_back(cell);
        pipeline_contrib_bar.tick();
    }
    pipeline_contrib_bar.finish();

    // -- Step 2: Refine octree (uses the overload that accepts
    //    pre-built initial contributors) --
    auto [all_cells, all_contributors] = refine_octree(
        std::move(initial_cells),
        std::move(initial_contributors),
        positions,
        smoothing_lengths,
        isovalue,
        static_cast<std::uint32_t>(max_depth),
        domain,
        static_cast<std::uint32_t>(base_resolution),
        static_cast<std::uint32_t>(minimum_usable_hermite_samples),
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold);

    // -- Step 3: Solve QEF vertices for active leaf cells --
    vertices = solve_all_leaf_vertices(
        all_cells, all_contributors, positions,
        smoothing_lengths, isovalue);

    Py_END_ALLOW_THREADS

    // Return (positions_Nx3, normals_Nx3) as NumPy arrays.
    return build_vertices_numpy_result(vertices);
}

/**
 * @brief Python binding for occupied-solid classification scaffolding.
 *
 * Returns a dictionary describing the leaf-wise occupied solid derived from
 * the current adaptive octree. This is internal scaffolding for the upcoming
 * minimum-thickness regularizer.
 */
static PyObject *classify_occupied_solid_py(
        PyObject *self, PyObject *args) {
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    unsigned int base_resolution = 0U;
    double isovalue = 0.0;
    unsigned int max_depth = 0U;
    unsigned int minimum_usable_hermite_samples = 3U;
    double max_qef_rms_residual_ratio = 0.1;
    double min_normal_alignment_threshold = 0.97;
    double max_surface_leaf_size = 0.0;
    double erosion_radius = 0.0;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOOOIdI|Idddd",
                          &positions_object,
                          &smoothing_object,
                          &domain_min_object,
                          &domain_max_object,
                          &base_resolution,
                          &isovalue,
                          &max_depth,
                          &minimum_usable_hermite_samples,
                          &max_qef_rms_residual_ratio,
                          &min_normal_alignment_threshold,
                          &max_surface_leaf_size,
                          &erosion_radius)) {
        return NULL;
    }

    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    BoundingBox domain{};

    if (!parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths must match in size");
        return NULL;
    }
    if (!parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max)) {
        return NULL;
    }

    std::vector<OccupiedSolidLeaf> solid_leaves;
    std::vector<double> clearance;
    std::vector<std::uint8_t> eroded_inside;
    std::vector<double> dilation_distance;
    std::vector<std::uint8_t> opened_inside;
    std::vector<OpenedBoundarySample> opened_boundary_samples;
    OpenedSurfaceMesh opened_surface_mesh;
    std::size_t inside_count = 0U;
    std::size_t boundary_inside_count = 0U;
    std::size_t boundary_outside_count = 0U;
    std::size_t eroded_inside_count = 0U;
    std::size_t opened_inside_count = 0U;

    Py_BEGIN_ALLOW_THREADS

    TopLevelParticleGrid grid(domain, base_resolution);
    grid.insert_particles(positions);
    grid.compute_bin_max_h(smoothing_lengths);

    std::vector<OctreeCell> top_cells =
        create_top_level_cells(domain, base_resolution);
    std::vector<OctreeCell> initial_cells;
    initial_cells.reserve(top_cells.size());
    std::vector<std::size_t> initial_contributors;

    for (std::size_t ci = 0; ci < top_cells.size(); ++ci) {
        OctreeCell cell = top_cells[ci];
        std::uint32_t sx = 0, sy = 0, sz = 0;
        std::uint32_t ex = 0, ey = 0, ez = 0;
        grid.contributor_bin_span(
            cell.bounds, smoothing_lengths, sx, sy, sz, ex, ey, ez);

        const std::int64_t begin =
            static_cast<std::int64_t>(initial_contributors.size());
        for (std::uint32_t ix = sx; ix <= ex; ++ix) {
            for (std::uint32_t iy = sy; iy <= ey; ++iy) {
                for (std::uint32_t iz = sz; iz <= ez; ++iz) {
                    const TopLevelBin &bin =
                        grid.bins[grid.flatten_index(ix, iy, iz)];
                    for (std::size_t pi : bin.particle_indices) {
                        if (particle_support_overlaps_box(
                                positions[pi], smoothing_lengths[pi],
                                cell.bounds)) {
                            initial_contributors.push_back(pi);
                        }
                    }
                }
            }
        }

        const std::int64_t end =
            static_cast<std::int64_t>(initial_contributors.size());
        cell.contributor_begin = begin;
        cell.contributor_end = end;
        initial_cells.push_back(cell);
    }

    auto [all_cells, all_contributors] = refine_octree(
        std::move(initial_cells), std::move(initial_contributors),
        positions, smoothing_lengths, isovalue,
        static_cast<std::uint32_t>(max_depth), domain,
        static_cast<std::uint32_t>(base_resolution),
        static_cast<std::uint32_t>(minimum_usable_hermite_samples),
        max_qef_rms_residual_ratio, min_normal_alignment_threshold);

    while (refine_surface_band_cells(
        all_cells, all_contributors, positions, smoothing_lengths,
        isovalue, static_cast<std::uint32_t>(max_depth),
        max_surface_leaf_size,
        static_cast<std::uint32_t>(minimum_usable_hermite_samples),
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold)) {
    }

    balance_octree(
        all_cells, all_contributors, positions, smoothing_lengths,
        isovalue, domain, static_cast<std::uint32_t>(base_resolution),
        static_cast<std::uint32_t>(max_depth));

    LeafSpatialIndex spatial_index;
    spatial_index.build(
        all_cells, domain, static_cast<std::uint32_t>(max_depth),
        static_cast<std::uint32_t>(base_resolution));
    solid_leaves = classify_occupied_solid_leaves(
        all_cells, all_contributors, positions, smoothing_lengths,
        spatial_index, isovalue, static_cast<std::uint32_t>(max_depth));
    clearance = compute_inside_clearance(solid_leaves);
    eroded_inside = erode_occupied_solid_leaves(
        solid_leaves, clearance, erosion_radius);
    dilation_distance = compute_distance_to_eroded_solid(
        solid_leaves, eroded_inside);
    opened_inside = dilate_eroded_solid_leaves(
        eroded_inside, dilation_distance, erosion_radius);
    opened_boundary_samples = generate_opened_boundary_samples(
        solid_leaves, opened_inside, all_cells);
    opened_surface_mesh = generate_opened_surface_mesh(
        solid_leaves, opened_inside, all_cells, spatial_index,
        domain, static_cast<std::uint32_t>(base_resolution),
        static_cast<std::uint32_t>(max_depth));

    for (std::size_t i = 0; i < solid_leaves.size(); ++i) {
        const OccupiedSolidLeaf &leaf = solid_leaves[i];
        if (leaf.occupancy == OccupancyState::kInside ||
            leaf.occupancy == OccupancyState::kBoundaryInside) {
            ++inside_count;
        }
        if (leaf.occupancy == OccupancyState::kBoundaryInside) {
            ++boundary_inside_count;
        }
        if (leaf.occupancy == OccupancyState::kBoundaryOutside) {
            ++boundary_outside_count;
        }
        if (eroded_inside[i] != 0U) {
            ++eroded_inside_count;
        }
        if (opened_inside[i] != 0U) {
            ++opened_inside_count;
        }
    }

    Py_END_ALLOW_THREADS

    const npy_intp dims[1] = {static_cast<npy_intp>(solid_leaves.size())};
    PyObject *occupancy_array = PyArray_SimpleNew(1, dims, NPY_UINT8);
    PyObject *depth_array = PyArray_SimpleNew(1, dims, NPY_UINT32);
    PyObject *center_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *size_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *clearance_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *eroded_inside_array = PyArray_SimpleNew(1, dims, NPY_UINT8);
    PyObject *dilation_distance_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *opened_inside_array = PyArray_SimpleNew(1, dims, NPY_UINT8);
    const npy_intp sample_dims[2] = {
        static_cast<npy_intp>(opened_boundary_samples.size()), 3};
    PyObject *sample_positions_array =
        PyArray_SimpleNew(2, sample_dims, NPY_DOUBLE);
    PyObject *sample_normals_array =
        PyArray_SimpleNew(2, sample_dims, NPY_DOUBLE);
    const npy_intp mesh_vertex_dims[2] = {
        static_cast<npy_intp>(opened_surface_mesh.vertices.size()), 3};
    const npy_intp mesh_face_dims[2] = {
        static_cast<npy_intp>(opened_surface_mesh.triangles.size()), 3};
    PyObject *mesh_vertex_array =
        PyArray_SimpleNew(2, mesh_vertex_dims, NPY_DOUBLE);
    PyObject *mesh_face_array =
        PyArray_SimpleNew(2, mesh_face_dims, NPY_UINT32);
    if (!occupancy_array || !depth_array || !center_array || !size_array ||
        !clearance_array || !eroded_inside_array ||
        !dilation_distance_array || !opened_inside_array ||
        !sample_positions_array || !sample_normals_array ||
        !mesh_vertex_array || !mesh_face_array) {
        Py_XDECREF(occupancy_array);
        Py_XDECREF(depth_array);
        Py_XDECREF(center_array);
        Py_XDECREF(size_array);
        Py_XDECREF(clearance_array);
        Py_XDECREF(eroded_inside_array);
        Py_XDECREF(dilation_distance_array);
        Py_XDECREF(opened_inside_array);
        Py_XDECREF(sample_positions_array);
        Py_XDECREF(sample_normals_array);
        Py_XDECREF(mesh_vertex_array);
        Py_XDECREF(mesh_face_array);
        return NULL;
    }

    auto *occupancy_data = static_cast<std::uint8_t *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(occupancy_array)));
    auto *depth_data = static_cast<std::uint32_t *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(depth_array)));
    auto *center_data = static_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(center_array)));
    auto *size_data = static_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(size_array)));
    auto *clearance_data = static_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(clearance_array)));
    auto *eroded_inside_data = static_cast<std::uint8_t *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(eroded_inside_array)));
    auto *dilation_distance_data = static_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(dilation_distance_array)));
    auto *opened_inside_data = static_cast<std::uint8_t *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(opened_inside_array)));
    auto *sample_positions_data = static_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(sample_positions_array)));
    auto *sample_normals_data = static_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(sample_normals_array)));
    auto *mesh_vertex_data = static_cast<double *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(mesh_vertex_array)));
    auto *mesh_face_data = static_cast<std::uint32_t *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(mesh_face_array)));

    for (std::size_t i = 0; i < solid_leaves.size(); ++i) {
        occupancy_data[i] = static_cast<std::uint8_t>(solid_leaves[i].occupancy);
        depth_data[i] = solid_leaves[i].depth;
        center_data[i] = solid_leaves[i].center_value;
        size_data[i] = solid_leaves[i].cell_size;
        clearance_data[i] = clearance[i];
        eroded_inside_data[i] = eroded_inside[i];
        dilation_distance_data[i] = dilation_distance[i];
        opened_inside_data[i] = opened_inside[i];
    }

    for (std::size_t i = 0; i < opened_boundary_samples.size(); ++i) {
        sample_positions_data[i * 3] = opened_boundary_samples[i].position.x;
        sample_positions_data[i * 3 + 1] = opened_boundary_samples[i].position.y;
        sample_positions_data[i * 3 + 2] = opened_boundary_samples[i].position.z;
        sample_normals_data[i * 3] = opened_boundary_samples[i].outward_normal.x;
        sample_normals_data[i * 3 + 1] = opened_boundary_samples[i].outward_normal.y;
        sample_normals_data[i * 3 + 2] = opened_boundary_samples[i].outward_normal.z;
    }

    for (std::size_t i = 0; i < opened_surface_mesh.vertices.size(); ++i) {
        mesh_vertex_data[i * 3] = opened_surface_mesh.vertices[i].position.x;
        mesh_vertex_data[i * 3 + 1] = opened_surface_mesh.vertices[i].position.y;
        mesh_vertex_data[i * 3 + 2] = opened_surface_mesh.vertices[i].position.z;
    }
    for (std::size_t i = 0; i < opened_surface_mesh.triangles.size(); ++i) {
        mesh_face_data[i * 3] = opened_surface_mesh.triangles[i].vertex_indices[0];
        mesh_face_data[i * 3 + 1] = opened_surface_mesh.triangles[i].vertex_indices[1];
        mesh_face_data[i * 3 + 2] = opened_surface_mesh.triangles[i].vertex_indices[2];
    }

    return Py_BuildValue(
        "{s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n}",
        "occupancy", occupancy_array,
        "depths", depth_array,
        "center_values", center_array,
        "cell_sizes", size_array,
        "clearance", clearance_array,
        "eroded_inside", eroded_inside_array,
        "dilation_distance", dilation_distance_array,
        "opened_inside", opened_inside_array,
        "opened_boundary_positions", sample_positions_array,
        "opened_boundary_normals", sample_normals_array,
        "opened_surface_vertices", mesh_vertex_array,
        "opened_surface_faces", mesh_face_array,
        "n_leaves", static_cast<Py_ssize_t>(solid_leaves.size()),
        "n_inside", static_cast<Py_ssize_t>(inside_count),
        "n_boundary_inside", static_cast<Py_ssize_t>(boundary_inside_count),
        "n_boundary_outside", static_cast<Py_ssize_t>(boundary_outside_count),
        "n_eroded_inside", static_cast<Py_ssize_t>(eroded_inside_count),
        "n_opened_inside", static_cast<Py_ssize_t>(opened_inside_count),
        "n_opened_boundary_samples",
        static_cast<Py_ssize_t>(opened_boundary_samples.size()),
        "n_opened_surface_vertices",
        static_cast<Py_ssize_t>(opened_surface_mesh.vertices.size()),
        "n_opened_surface_faces",
        static_cast<Py_ssize_t>(opened_surface_mesh.triangles.size()));
}

/**
 * @brief Python binding for friends-of-friends clustering.
 *
 * Args (positional):
 *     positions: (N, 3) float64 array of point positions.
 *     domain_min: 3-tuple giving the domain lower corner.
 *     domain_max: 3-tuple giving the domain upper corner.
 *     linking_factor: float scaling factor for the linking length.
 *
 * Returns:
 *     1-D int64 NumPy array of length N with group labels (0-based).
 */
static PyObject *fof_cluster_py(PyObject * /* self */,
                                PyObject *args) {
    PyObject *positions_object = NULL;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    double linking_factor = 1.5;

    if (!PyArg_ParseTuple(args, "OOOd",
                          &positions_object,
                          &domain_min_object,
                          &domain_max_object,
                          &linking_factor)) {
        return NULL;
    }

    // Parse positions via the fast buffer path, falling back to the
    // sequence parser if the buffer protocol is not supported.
    std::vector<Vector3d> positions;
    if (!try_parse_positions_buffer(positions_object, positions)) {
        if (!parse_vector3d_sequence(positions_object, positions)) {
            PyErr_SetString(
                PyExc_TypeError,
                "positions must be an (N, 3) float64 array or "
                "sequence of 3-tuples");
            return NULL;
        }
    }

    // Parse domain bounds.
    Vector3d domain_min, domain_max;
    if (!parse_vector3d(domain_min_object, domain_min) ||
        !parse_vector3d(domain_max_object, domain_max)) {
        return NULL;
    }

    // Run FOF clustering with the GIL released.
    std::vector<std::int64_t> labels;

    Py_BEGIN_ALLOW_THREADS
    labels = fof_cluster(positions, domain_min, domain_max,
                         linking_factor);
    Py_END_ALLOW_THREADS

    // Build a 1-D int64 NumPy array from the labels.
    const npy_intp dims[1] = {
        static_cast<npy_intp>(labels.size())};
    PyObject *result = PyArray_SimpleNew(1, dims, NPY_INT64);
    if (result == NULL) {
        return NULL;
    }

    // Copy label data into the NumPy array.
    void *dest = PyArray_DATA(
        reinterpret_cast<PyArrayObject *>(result));
    std::memcpy(dest, labels.data(),
                labels.size() * sizeof(std::int64_t));

    return result;
}

/**
 * @brief Python binding for the full particles-to-mesh pipeline.
 *
 * Args (positional):
 *   positions         – (N, 3) float64 array
 *   smoothing_lengths – (N,) float64 array
 *   domain_min        – 3-tuple
 *   domain_max        – 3-tuple
 *   base_resolution   – unsigned int
 *   isovalue          – double
 *   max_depth         – unsigned int
 *   smoothing_iterations – unsigned int (default 0)
 *   smoothing_strength   – double (default 0.5)
 *   max_edge_ratio       – double (default 1.5)
 *
 * Returns a dict with keys:
 *   vertices            – (V, 3) float64 ndarray
 *   faces               – (F, 3) uint32 ndarray
 *   isovalue            – float
 *   n_qef_vertices      – int
 */
static PyObject *run_full_pipeline_py(
        PyObject * /*self*/, PyObject *args) {
    PyObject *positions_object = NULL;
    PyObject *smoothing_object = NULL;
    PyObject *domain_min_object = NULL;
    PyObject *domain_max_object = NULL;
    unsigned int base_resolution = 0U;
    double isovalue = 0.0;
    unsigned int max_depth = 0U;
    unsigned int smoothing_iterations = 0U;
    double smoothing_strength = 0.5;
    double max_edge_ratio = 1.5;
    unsigned int minimum_usable_hermite_samples = 3U;
    double max_qef_rms_residual_ratio = 0.1;
    double min_normal_alignment_threshold = 0.97;
    double min_feature_thickness = 0.0;

    if (!PyArg_ParseTuple(args, "OOOOIdI|IddIddd",
                          &positions_object,
                          &smoothing_object,
                          &domain_min_object,
                          &domain_max_object,
                          &base_resolution,
                          &isovalue,
                          &max_depth,
                          &smoothing_iterations,
                          &smoothing_strength,
                          &max_edge_ratio,
                          &minimum_usable_hermite_samples,
                          &max_qef_rms_residual_ratio,
                          &min_normal_alignment_threshold,
                          &min_feature_thickness)) {
        return NULL;
    }

    // Parse particle data.
    std::vector<Vector3d> positions;
    std::vector<double> smoothing_lengths;
    BoundingBox domain{};

    if (!parse_positions(positions_object, positions) ||
        !parse_doubles(smoothing_object, smoothing_lengths)) {
        return NULL;
    }
    if (positions.size() != smoothing_lengths.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "positions and smoothing lengths "
                        "must match in size");
        return NULL;
    }
    if (!parse_vector3d(domain_min_object, domain.min) ||
        !parse_vector3d(domain_max_object, domain.max)) {
        return NULL;
    }
    if (base_resolution == 0U) {
        PyErr_SetString(PyExc_ValueError,
                        "base_resolution must be > 0");
        return NULL;
    }

    // Release GIL and run the entire pipeline in C++.
    DCPipelineResult result;

    Py_BEGIN_ALLOW_THREADS

    result = run_dc_pipeline(
        positions, smoothing_lengths, domain,
        static_cast<std::uint32_t>(base_resolution),
        isovalue,
        static_cast<std::uint32_t>(max_depth),
        static_cast<std::uint32_t>(smoothing_iterations),
        smoothing_strength,
        max_edge_ratio,
        static_cast<std::uint32_t>(minimum_usable_hermite_samples),
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
        min_feature_thickness);

    Py_END_ALLOW_THREADS

    // Build output NumPy arrays.
    const std::size_t nv = result.vertices.size();
    const std::size_t nf = result.triangles.size();

    npy_intp vdims[2] = {
        static_cast<npy_intp>(nv), 3};
    PyObject *v_arr =
        PyArray_SimpleNew(2, vdims, NPY_DOUBLE);
    if (v_arr == NULL) return NULL;
    double *v_data = static_cast<double *>(
        PyArray_DATA(
            reinterpret_cast<PyArrayObject *>(v_arr)));
    for (std::size_t i = 0; i < nv; ++i) {
        v_data[i * 3] = result.vertices[i].x;
        v_data[i * 3 + 1] = result.vertices[i].y;
        v_data[i * 3 + 2] = result.vertices[i].z;
    }

    npy_intp fdims[2] = {
        static_cast<npy_intp>(nf), 3};
    PyObject *f_arr =
        PyArray_SimpleNew(2, fdims, NPY_UINT32);
    if (f_arr == NULL) {
        Py_DECREF(v_arr);
        return NULL;
    }
    std::uint32_t *f_data = static_cast<std::uint32_t *>(
        PyArray_DATA(
            reinterpret_cast<PyArrayObject *>(f_arr)));
    for (std::size_t i = 0; i < nf; ++i) {
        f_data[i * 3] = result.triangles[i][0];
        f_data[i * 3 + 1] = result.triangles[i][1];
        f_data[i * 3 + 2] = result.triangles[i][2];
    }

    // Build result dict.
    PyObject *dict = PyDict_New();
    if (dict == NULL) {
        Py_DECREF(v_arr);
        Py_DECREF(f_arr);
        return NULL;
    }
    PyDict_SetItemString(dict, "vertices", v_arr);
    PyDict_SetItemString(dict, "faces", f_arr);
    Py_DECREF(v_arr);
    Py_DECREF(f_arr);
    PyDict_SetItemString(dict, "isovalue",
        PyFloat_FromDouble(result.isovalue));
    PyDict_SetItemString(dict, "n_qef_vertices",
        PyLong_FromSize_t(result.n_qef_vertices));
    return dict;
}

/**
 * @brief Python methods exported by the adaptive extension.
 */
static PyMethodDef adaptive_methods[] = {
    {
        "fof_cluster",
        fof_cluster_py,
        METH_VARARGS,
        PyDoc_STR(
            "Run FOF clustering on vertex positions."),
    },
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
        "create_top_level_cells_with_contributors",
        create_top_level_cells_with_contributors_py,
        METH_VARARGS,
        PyDoc_STR("Create top-level cells with contributors in one pass."),
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
        "build_refined_tree",
        build_refined_tree_py,
        METH_VARARGS,
        PyDoc_STR(
            "Build and refine the adaptive octree in C++ and return "
            "(cells, contributors) for resumable Python workflows."),
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
    {
        "refine_octree",
        refine_octree_py,
        METH_VARARGS,
        PyDoc_STR("Refine the octree using breadth-first refinement."),
    },
    {
        "hermite_samples_for_cell",
        hermite_samples_for_cell_py,
        METH_VARARGS,
        PyDoc_STR("Compute Hermite samples for one leaf cell."),
    },
    {
        "solve_qef_for_leaf",
        solve_qef_for_leaf_py,
        METH_VARARGS,
        PyDoc_STR("Solve the QEF and return the representative vertex for a leaf cell."),
    },
    {
        "generate_mesh",
        generate_mesh_py,
        METH_VARARGS,
        PyDoc_STR("Generate a triangle mesh from a refined octree via dual contouring."),
    },
    {
        "solve_vertices",
        solve_vertices_py,
        METH_VARARGS,
        PyDoc_STR("Solve QEF vertices for active leaf cells (no face generation)."),
    },
    {
        "set_num_threads",
        [](PyObject *, PyObject *args) -> PyObject * {
            int n;
            if (!PyArg_ParseTuple(args, "i", &n)) {
                return NULL;
            }
            if (n < 1) {
                PyErr_SetString(PyExc_ValueError,
                                "nthreads must be >= 1");
                return NULL;
            }
            omp_set_num_threads(n);
            Py_RETURN_NONE;
        },
        METH_VARARGS,
        PyDoc_STR("Set the number of OpenMP threads."),
    },
    {
        "run_octree_pipeline",
        run_octree_pipeline_py,
        METH_VARARGS,
        PyDoc_STR(
            "Run the octree pipeline in C++ "
            "(build + refine + solve vertices) "
            "and return (positions, normals)."),
    },
    {
        "classify_occupied_solid",
        classify_occupied_solid_py,
        METH_VARARGS,
        PyDoc_STR(
            "Classify the adaptive occupied solid for morphology scaffolding."),
    },
    {
        "run_full_pipeline",
        run_full_pipeline_py,
        METH_VARARGS,
        PyDoc_STR(
            "Run the full particles-to-mesh pipeline "
            "(adaptive octree + reconstruction) in C++ "
            "and return a dict with vertices, faces, and "
            "basic pipeline metadata."),
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
PyMODINIT_FUNC PyInit__adaptive(void) {
    import_array();  /* Initialize NumPy C API. */
    return PyModule_Create(&adaptive_module);
}
