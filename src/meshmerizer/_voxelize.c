#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdio.h>

static npy_int64 flatten_chunk_index(
    npy_int64 ix,
    npy_int64 iy,
    npy_int64 iz,
    npy_int64 nchunks
) {
    return ((ix * nchunks) + iy) * nchunks + iz;
}

static int deposit_box_kernel(
    npy_float64 *grid,
    npy_intp grid_size,
    npy_intp n_points,
    npy_float64 *data,
    npy_int64 *coords,
    npy_int64 *smoothing,
    npy_int64 nx,
    npy_int64 ny,
    npy_int64 nz
) {
    for (npy_intp idx = 0; idx < grid_size; idx++) {
        grid[idx] = 0.0;
    }

    // The deposition helper is intentionally serial. The chunk pipeline
    // parallelizes at a higher level, so this helper only needs one code path.
    for (npy_intp i = 0; i < n_points; i++) {
        npy_int64 cx = coords[i * 3 + 0];
        npy_int64 cy = coords[i * 3 + 1];
        npy_int64 cz = coords[i * 3 + 2];
        npy_int64 h = smoothing[i];
        npy_float64 val = data[i];

        if (h < 0) {
            h = 0;
        }

        npy_int64 ii_start = cx - h;
        npy_int64 ii_end = cx + h;
        npy_int64 jj_start = cy - h;
        npy_int64 jj_end = cy + h;
        npy_int64 kk_start = cz - h;
        npy_int64 kk_end = cz + h;

        for (npy_int64 ii = ii_start; ii <= ii_end; ii++) {
            for (npy_int64 jj = jj_start; jj <= jj_end; jj++) {
                for (npy_int64 kk = kk_start; kk <= kk_end; kk++) {
                    if (ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz) {
                        npy_intp idx = ii * (ny * nz) + jj * nz + kk;
                        grid[idx] += val;
                    }
                }
            }
        }
    }
    return 0;
}

static PyObject* voxelize_box_deposition(PyObject* self, PyObject* args) {
    PyArrayObject *grid_arr_in = NULL, *data_arr_in = NULL, *coords_arr_in = NULL, *smoothing_arr_in = NULL;
    PyArrayObject *grid_arr = NULL, *data_arr = NULL, *coords_arr = NULL, *smoothing_arr = NULL;
    int resolution;

    // Parse arguments as Python Objects first
    if (!PyArg_ParseTuple(args, "O!O!O!O!i",
        &PyArray_Type, &grid_arr_in,
        &PyArray_Type, &data_arr_in,
        &PyArray_Type, &coords_arr_in,
        &PyArray_Type, &smoothing_arr_in,
        &resolution)) {
        return NULL;
    }

    // Force arrays to be contiguous, aligned, and of the correct type
    // NPY_ARRAY_CARRAY = C_CONTIGUOUS | ALIGNED | WRITEABLE (if needed)
    // For inputs, we ideally want NPY_ARRAY_CARRAY_RO, but CARRAY is fine as we won't write to read-only inputs unless we have to.
    // Actually, PyArray_FROM_OTF with NPY_ARRAY_CARRAY will copy if necessary.
    
    // Grid: Read-Write, Float64
    grid_arr = (PyArrayObject *)PyArray_FROM_OTF((PyObject*)grid_arr_in, NPY_FLOAT64, NPY_ARRAY_CARRAY);
    // Data: Read-Only (effectively), Float64
    data_arr = (PyArrayObject *)PyArray_FROM_OTF((PyObject*)data_arr_in, NPY_FLOAT64, NPY_ARRAY_CARRAY);
    // Coords: Read-Only, Int64
    coords_arr = (PyArrayObject *)PyArray_FROM_OTF((PyObject*)coords_arr_in, NPY_INT64, NPY_ARRAY_CARRAY);
    // Smoothing: Read-Only, Int64
    smoothing_arr = (PyArrayObject *)PyArray_FROM_OTF((PyObject*)smoothing_arr_in, NPY_INT64, NPY_ARRAY_CARRAY);

    if (!grid_arr || !data_arr || !coords_arr || !smoothing_arr) {
        Py_XDECREF(grid_arr);
        Py_XDECREF(data_arr);
        Py_XDECREF(coords_arr);
        Py_XDECREF(smoothing_arr);
        PyErr_SetString(PyExc_RuntimeError, "Could not convert input arrays to contiguous C-arrays.");
        return NULL;
    }

    // Check Dimensions
    npy_intp n_points = PyArray_DIM(data_arr, 0);
    
    // Check coords shape: (N, 3)
    if (PyArray_NDIM(coords_arr) != 2 || PyArray_DIM(coords_arr, 0) != n_points || PyArray_DIM(coords_arr, 1) != 3) {
        Py_DECREF(grid_arr); Py_DECREF(data_arr); Py_DECREF(coords_arr); Py_DECREF(smoothing_arr);
        PyErr_Format(PyExc_ValueError, "Coordinates must be (N, 3). Got (%ld, %ld)", 
                     (long)PyArray_DIM(coords_arr, 0), (long)(PyArray_NDIM(coords_arr) > 1 ? PyArray_DIM(coords_arr, 1) : 0));
        return NULL;
    }
    
    // Check smoothing shape: (N,) or (N, 1)? Assuming (N,) from numpy
    if (PyArray_DIM(smoothing_arr, 0) != n_points) {
        Py_DECREF(grid_arr); Py_DECREF(data_arr); Py_DECREF(coords_arr); Py_DECREF(smoothing_arr);
        PyErr_Format(PyExc_ValueError, "Smoothing lengths must match data length %ld.", (long)n_points);
        return NULL;
    }

    // Pointers to data
    npy_float64 *grid = (npy_float64*)PyArray_DATA(grid_arr);
    npy_float64 *data = (npy_float64*)PyArray_DATA(data_arr);
    npy_int64 *coords = (npy_int64*)PyArray_DATA(coords_arr);
    npy_int64 *smoothing = (npy_int64*)PyArray_DATA(smoothing_arr);
    npy_intp grid_size = PyArray_SIZE(grid_arr);

    if (deposit_box_kernel(
        grid,
        grid_size,
        n_points,
        data,
        coords,
        smoothing,
        resolution,
        resolution,
        resolution
    ) != 0) {
        Py_DECREF(grid_arr);
        Py_DECREF(data_arr);
        Py_DECREF(coords_arr);
        Py_DECREF(smoothing_arr);
        PyErr_SetString(PyExc_MemoryError, "Could not allocate thread-local voxel grids.");
        return NULL;
    }

    // Clean up references
    Py_DECREF(grid_arr);
    Py_DECREF(data_arr);
    Py_DECREF(coords_arr);
    Py_DECREF(smoothing_arr);

    Py_RETURN_NONE;
}

static PyObject* voxelize_box_deposition_local(PyObject* self, PyObject* args) {
    PyArrayObject *grid_arr_in = NULL, *data_arr_in = NULL, *coords_arr_in = NULL, *smoothing_arr_in = NULL;
    PyArrayObject *grid_arr = NULL, *data_arr = NULL, *coords_arr = NULL, *smoothing_arr = NULL;
    int nx, ny, nz;

    if (!PyArg_ParseTuple(args, "O!O!O!O!iii",
        &PyArray_Type, &grid_arr_in,
        &PyArray_Type, &data_arr_in,
        &PyArray_Type, &coords_arr_in,
        &PyArray_Type, &smoothing_arr_in,
        &nx,
        &ny,
        &nz)) {
        return NULL;
    }

    grid_arr = (PyArrayObject *)PyArray_FROM_OTF((PyObject*)grid_arr_in, NPY_FLOAT64, NPY_ARRAY_CARRAY);
    data_arr = (PyArrayObject *)PyArray_FROM_OTF((PyObject*)data_arr_in, NPY_FLOAT64, NPY_ARRAY_CARRAY);
    coords_arr = (PyArrayObject *)PyArray_FROM_OTF((PyObject*)coords_arr_in, NPY_INT64, NPY_ARRAY_CARRAY);
    smoothing_arr = (PyArrayObject *)PyArray_FROM_OTF((PyObject*)smoothing_arr_in, NPY_INT64, NPY_ARRAY_CARRAY);

    if (!grid_arr || !data_arr || !coords_arr || !smoothing_arr) {
        Py_XDECREF(grid_arr);
        Py_XDECREF(data_arr);
        Py_XDECREF(coords_arr);
        Py_XDECREF(smoothing_arr);
        PyErr_SetString(PyExc_RuntimeError, "Could not convert input arrays to contiguous C-arrays.");
        return NULL;
    }

    npy_intp n_points = PyArray_DIM(data_arr, 0);
    if (PyArray_NDIM(coords_arr) != 2 || PyArray_DIM(coords_arr, 0) != n_points || PyArray_DIM(coords_arr, 1) != 3) {
        Py_DECREF(grid_arr); Py_DECREF(data_arr); Py_DECREF(coords_arr); Py_DECREF(smoothing_arr);
        PyErr_Format(PyExc_ValueError, "Coordinates must be (N, 3). Got (%ld, %ld)",
                     (long)PyArray_DIM(coords_arr, 0), (long)(PyArray_NDIM(coords_arr) > 1 ? PyArray_DIM(coords_arr, 1) : 0));
        return NULL;
    }
    if (PyArray_DIM(smoothing_arr, 0) != n_points) {
        Py_DECREF(grid_arr); Py_DECREF(data_arr); Py_DECREF(coords_arr); Py_DECREF(smoothing_arr);
        PyErr_Format(PyExc_ValueError, "Smoothing lengths must match data length %ld.", (long)n_points);
        return NULL;
    }
    if (PyArray_NDIM(grid_arr) != 3 ||
        PyArray_DIM(grid_arr, 0) != nx ||
        PyArray_DIM(grid_arr, 1) != ny ||
        PyArray_DIM(grid_arr, 2) != nz) {
        Py_DECREF(grid_arr); Py_DECREF(data_arr); Py_DECREF(coords_arr); Py_DECREF(smoothing_arr);
        PyErr_SetString(PyExc_ValueError, "Grid shape must match the provided local dimensions.");
        return NULL;
    }

    npy_float64 *grid = (npy_float64*)PyArray_DATA(grid_arr);
    npy_float64 *data = (npy_float64*)PyArray_DATA(data_arr);
    npy_int64 *coords = (npy_int64*)PyArray_DATA(coords_arr);
    npy_int64 *smoothing = (npy_int64*)PyArray_DATA(smoothing_arr);
    npy_intp grid_size = PyArray_SIZE(grid_arr);

    if (deposit_box_kernel(
        grid,
        grid_size,
        n_points,
        data,
        coords,
        smoothing,
        (npy_int64)nx,
        (npy_int64)ny,
        (npy_int64)nz
    ) != 0) {
        Py_DECREF(grid_arr);
        Py_DECREF(data_arr);
        Py_DECREF(coords_arr);
        Py_DECREF(smoothing_arr);
        PyErr_SetString(PyExc_MemoryError, "Could not allocate thread-local voxel grids.");
        return NULL;
    }

    Py_DECREF(grid_arr);
    Py_DECREF(data_arr);
    Py_DECREF(coords_arr);
    Py_DECREF(smoothing_arr);

    Py_RETURN_NONE;
}

static PyObject* voxelize_build_chunk_particle_index(
    PyObject* self,
    PyObject* args
) {
    PyArrayObject *coords_arr_in = NULL;
    PyArrayObject *radius_arr_in = NULL;
    PyArrayObject *lower_arr_in = NULL;
    PyArrayObject *upper_arr_in = NULL;
    PyArrayObject *coords_arr = NULL;
    PyArrayObject *radius_arr = NULL;
    PyArrayObject *lower_arr = NULL;
    PyArrayObject *upper_arr = NULL;
    int nchunks;

    if (!PyArg_ParseTuple(
        args,
        "O!O!O!O!i",
        &PyArray_Type,
        &coords_arr_in,
        &PyArray_Type,
        &radius_arr_in,
        &PyArray_Type,
        &lower_arr_in,
        &PyArray_Type,
        &upper_arr_in,
        &nchunks
    )) {
        return NULL;
    }

    coords_arr = (PyArrayObject *)PyArray_FROM_OTF(
        (PyObject*)coords_arr_in,
        NPY_FLOAT64,
        NPY_ARRAY_CARRAY
    );
    radius_arr = (PyArrayObject *)PyArray_FROM_OTF(
        (PyObject*)radius_arr_in,
        NPY_FLOAT64,
        NPY_ARRAY_CARRAY
    );
    lower_arr = (PyArrayObject *)PyArray_FROM_OTF(
        (PyObject*)lower_arr_in,
        NPY_FLOAT64,
        NPY_ARRAY_CARRAY
    );
    upper_arr = (PyArrayObject *)PyArray_FROM_OTF(
        (PyObject*)upper_arr_in,
        NPY_FLOAT64,
        NPY_ARRAY_CARRAY
    );

    if (!coords_arr || !radius_arr || !lower_arr || !upper_arr) {
        Py_XDECREF(coords_arr);
        Py_XDECREF(radius_arr);
        Py_XDECREF(lower_arr);
        Py_XDECREF(upper_arr);
        PyErr_SetString(
            PyExc_RuntimeError,
            "Could not convert chunk-index inputs to contiguous arrays."
        );
        return NULL;
    }

    if (nchunks < 1) {
        Py_DECREF(coords_arr);
        Py_DECREF(radius_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        PyErr_SetString(PyExc_ValueError, "nchunks must be >= 1");
        return NULL;
    }

    npy_intp n_points = PyArray_DIM(coords_arr, 0);
    if (PyArray_NDIM(coords_arr) != 2 ||
        PyArray_DIM(coords_arr, 1) != 3) {
        Py_DECREF(coords_arr);
        Py_DECREF(radius_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        PyErr_SetString(PyExc_ValueError, "coordinates must have shape (N, 3)");
        return NULL;
    }
    if (PyArray_NDIM(radius_arr) != 1 || PyArray_DIM(radius_arr, 0) != n_points) {
        Py_DECREF(coords_arr);
        Py_DECREF(radius_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        PyErr_SetString(PyExc_ValueError, "support radii must have shape (N,)");
        return NULL;
    }
    if (PyArray_NDIM(lower_arr) != 1 ||
        PyArray_NDIM(upper_arr) != 1 ||
        PyArray_DIM(lower_arr, 0) != nchunks ||
        PyArray_DIM(upper_arr, 0) != nchunks) {
        Py_DECREF(coords_arr);
        Py_DECREF(radius_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        PyErr_SetString(
            PyExc_ValueError,
            "chunk bounds must have shape (nchunks,)"
        );
        return NULL;
    }

    npy_float64 *coords = (npy_float64*)PyArray_DATA(coords_arr);
    npy_float64 *radius = (npy_float64*)PyArray_DATA(radius_arr);
    npy_float64 *lower = (npy_float64*)PyArray_DATA(lower_arr);
    npy_float64 *upper = (npy_float64*)PyArray_DATA(upper_arr);
    npy_intp chunk_count = (npy_intp)nchunks * nchunks * nchunks;

    npy_intp *counts = (npy_intp*)calloc((size_t)chunk_count, sizeof(npy_intp));
    if (counts == NULL) {
        Py_DECREF(coords_arr);
        Py_DECREF(radius_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        PyErr_SetString(PyExc_MemoryError, "Could not allocate chunk counts.");
        return NULL;
    }

    for (npy_intp i = 0; i < n_points; i++) {
        npy_float64 min_x = coords[i * 3 + 0] - radius[i];
        npy_float64 min_y = coords[i * 3 + 1] - radius[i];
        npy_float64 min_z = coords[i * 3 + 2] - radius[i];
        npy_float64 max_x = coords[i * 3 + 0] + radius[i];
        npy_float64 max_y = coords[i * 3 + 1] + radius[i];
        npy_float64 max_z = coords[i * 3 + 2] + radius[i];

        npy_int64 start_x = 0;
        while (start_x < nchunks && upper[start_x] <= min_x) {
            start_x++;
        }
        npy_int64 start_y = 0;
        while (start_y < nchunks && upper[start_y] <= min_y) {
            start_y++;
        }
        npy_int64 start_z = 0;
        while (start_z < nchunks && upper[start_z] <= min_z) {
            start_z++;
        }

        npy_int64 stop_x = 0;
        while (stop_x < nchunks && lower[stop_x] <= max_x) {
            stop_x++;
        }
        npy_int64 stop_y = 0;
        while (stop_y < nchunks && lower[stop_y] <= max_y) {
            stop_y++;
        }
        npy_int64 stop_z = 0;
        while (stop_z < nchunks && lower[stop_z] <= max_z) {
            stop_z++;
        }

        if (start_x >= stop_x || start_y >= stop_y || start_z >= stop_z) {
            continue;
        }

        for (npy_int64 ix = start_x; ix < stop_x; ix++) {
            for (npy_int64 iy = start_y; iy < stop_y; iy++) {
                for (npy_int64 iz = start_z; iz < stop_z; iz++) {
                    npy_int64 flat_index = flatten_chunk_index(ix, iy, iz, nchunks);
                    counts[flat_index] += 1;
                }
            }
        }
    }

    npy_intp offsets_dims[1] = {chunk_count + 1};
    PyArrayObject *offsets_arr = (PyArrayObject *)PyArray_ZEROS(
        1,
        offsets_dims,
        NPY_INT64,
        0
    );
    if (offsets_arr == NULL) {
        free(counts);
        Py_DECREF(coords_arr);
        Py_DECREF(radius_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        return NULL;
    }

    npy_int64 *offsets = (npy_int64*)PyArray_DATA(offsets_arr);
    for (npy_intp idx = 0; idx < chunk_count; idx++) {
        offsets[idx + 1] = offsets[idx] + (npy_int64)counts[idx];
    }

    npy_intp payload_dims[1] = {offsets[chunk_count]};
    PyArrayObject *particle_arr = (PyArrayObject *)PyArray_EMPTY(
        1,
        payload_dims,
        NPY_INT64,
        0
    );
    if (particle_arr == NULL) {
        free(counts);
        Py_DECREF(offsets_arr);
        Py_DECREF(coords_arr);
        Py_DECREF(radius_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        return NULL;
    }

    npy_int64 *particle_indices = (npy_int64*)PyArray_DATA(particle_arr);
    npy_int64 *write_positions = (npy_int64*)malloc((size_t)chunk_count * sizeof(npy_int64));
    if (write_positions == NULL) {
        free(counts);
        Py_DECREF(offsets_arr);
        Py_DECREF(particle_arr);
        Py_DECREF(coords_arr);
        Py_DECREF(radius_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        PyErr_SetString(PyExc_MemoryError, "Could not allocate write positions.");
        return NULL;
    }
    for (npy_intp idx = 0; idx < chunk_count; idx++) {
        write_positions[idx] = offsets[idx];
    }

    for (npy_intp i = 0; i < n_points; i++) {
        npy_float64 min_x = coords[i * 3 + 0] - radius[i];
        npy_float64 min_y = coords[i * 3 + 1] - radius[i];
        npy_float64 min_z = coords[i * 3 + 2] - radius[i];
        npy_float64 max_x = coords[i * 3 + 0] + radius[i];
        npy_float64 max_y = coords[i * 3 + 1] + radius[i];
        npy_float64 max_z = coords[i * 3 + 2] + radius[i];

        npy_int64 start_x = 0;
        while (start_x < nchunks && upper[start_x] <= min_x) {
            start_x++;
        }
        npy_int64 start_y = 0;
        while (start_y < nchunks && upper[start_y] <= min_y) {
            start_y++;
        }
        npy_int64 start_z = 0;
        while (start_z < nchunks && upper[start_z] <= min_z) {
            start_z++;
        }

        npy_int64 stop_x = 0;
        while (stop_x < nchunks && lower[stop_x] <= max_x) {
            stop_x++;
        }
        npy_int64 stop_y = 0;
        while (stop_y < nchunks && lower[stop_y] <= max_y) {
            stop_y++;
        }
        npy_int64 stop_z = 0;
        while (stop_z < nchunks && lower[stop_z] <= max_z) {
            stop_z++;
        }

        if (start_x >= stop_x || start_y >= stop_y || start_z >= stop_z) {
            continue;
        }

        for (npy_int64 ix = start_x; ix < stop_x; ix++) {
            for (npy_int64 iy = start_y; iy < stop_y; iy++) {
                for (npy_int64 iz = start_z; iz < stop_z; iz++) {
                    npy_int64 flat_index = flatten_chunk_index(ix, iy, iz, nchunks);
                    particle_indices[write_positions[flat_index]++] = (npy_int64)i;
                }
            }
        }
    }

    free(write_positions);
    free(counts);
    Py_DECREF(coords_arr);
    Py_DECREF(radius_arr);
    Py_DECREF(lower_arr);
    Py_DECREF(upper_arr);

    return Py_BuildValue("NN", offsets_arr, particle_arr);
}

static PyMethodDef VoxelizeMethods[] = {
    {"box_deposition", voxelize_box_deposition, METH_VARARGS, "Voxelize with box deposition kernel."},
    {"box_deposition_local", voxelize_box_deposition_local, METH_VARARGS, "Voxelize with a local box deposition kernel."},
    {"build_chunk_particle_index", voxelize_build_chunk_particle_index, METH_VARARGS, "Build CSR-style chunk particle lists."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef voxelizemodule = {
    PyModuleDef_HEAD_INIT,
    "_voxelize",
    NULL,
    -1,
    VoxelizeMethods
};

PyMODINIT_FUNC PyInit__voxelize(void) {
    import_array();
    return PyModule_Create(&voxelizemodule);
}
