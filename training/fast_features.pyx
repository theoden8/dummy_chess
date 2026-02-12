# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Fast HalfKP feature extraction from PyArrow arrays.
Links directly to libdummychess.so and calls C++ NNUE feature extraction.
"""

import numpy
cimport numpy
cimport cython
from libc.stdint cimport int32_t, int64_t, uint8_t, uintptr_t

numpy.import_array()


# Declare the C++ function from NNUE.hpp
cdef extern from "../NNUE.hpp" namespace "preprocess":
    void extract_halfkp_features(
        const uint8_t* data,
        size_t length,
        bint flip,
        int32_t* white_indices,
        int32_t* black_indices,
        int* n_white,
        int* n_black,
        int64_t* stm
    ) nogil


def get_halfkp_features_arrow(fen_array, bint flip=False):
    """
    Extract HalfKP features from PyArrow BinaryArray.
    
    Args:
        fen_array: PyArrow BinaryArray or ChunkedArray of compressed FENs
        flip: If True, swap white/black perspectives
    
    Returns:
        Tuple of (white_indices, white_offsets, black_indices, black_offsets, stm)
        as numpy arrays, matching dummy_chess.get_halfkp_features_batch() output.
    """
    import pyarrow
    
    # Handle ChunkedArray by combining chunks
    if isinstance(fen_array, pyarrow.ChunkedArray):
        fen_array = fen_array.combine_chunks()
    
    cdef:
        Py_ssize_t n_positions = len(fen_array)
        Py_ssize_t i, j
        int n_white, n_black
        int64_t stm_val
        int32_t temp_white[32]
        int32_t temp_black[32]
        const uint8_t* data_ptr
        Py_ssize_t data_len
    
    # Get Arrow buffers directly (zero-copy)
    # BinaryArray layout: [validity_bitmap, offsets, data]
    buffers = fen_array.buffers()
    
    # offsets buffer - int32 offsets into data buffer
    offsets_buf = buffers[1]
    cdef const int32_t* offsets_ptr = <const int32_t*>(<uintptr_t>offsets_buf.address) if offsets_buf is not None else NULL
    
    # data buffer - raw bytes
    data_buf = buffers[2]
    cdef const uint8_t* all_data_ptr = <const uint8_t*>(<uintptr_t>data_buf.address) if data_buf is not None else NULL
    
    if offsets_ptr == NULL or all_data_ptr == NULL:
        raise ValueError("Invalid Arrow array buffers")
    
    # Pre-allocate output arrays (estimate ~16 pieces per position)
    cdef numpy.ndarray[int32_t, ndim=1] white_indices = numpy.empty(n_positions * 20, dtype=numpy.int32)
    cdef numpy.ndarray[int32_t, ndim=1] black_indices = numpy.empty(n_positions * 20, dtype=numpy.int32)
    cdef numpy.ndarray[int64_t, ndim=1] white_offsets = numpy.empty(n_positions, dtype=numpy.int64)
    cdef numpy.ndarray[int64_t, ndim=1] black_offsets = numpy.empty(n_positions, dtype=numpy.int64)
    cdef numpy.ndarray[int64_t, ndim=1] stm = numpy.empty(n_positions, dtype=numpy.int64)
    
    cdef int32_t* w_idx_ptr = <int32_t*>white_indices.data
    cdef int32_t* b_idx_ptr = <int32_t*>black_indices.data
    cdef int64_t* w_off_ptr = <int64_t*>white_offsets.data
    cdef int64_t* b_off_ptr = <int64_t*>black_offsets.data
    cdef int64_t* stm_ptr = <int64_t*>stm.data
    
    cdef Py_ssize_t w_total = 0, b_total = 0
    cdef int32_t start_offset, end_offset
    
    # Process each position using direct buffer access (no Python objects)
    with nogil:
        for i in range(n_positions):
            # Record offsets
            w_off_ptr[i] = w_total
            b_off_ptr[i] = b_total
            
            # Get data range from Arrow offsets
            start_offset = offsets_ptr[i]
            end_offset = offsets_ptr[i + 1]
            data_len = end_offset - start_offset
            
            if data_len == 0:
                stm_ptr[i] = 0
                continue
            
            data_ptr = all_data_ptr + start_offset
            
            # Call C++ feature extraction
            extract_halfkp_features(
                data_ptr, data_len, flip,
                temp_white, temp_black,
                &n_white, &n_black,
                &stm_val
            )
            
            stm_ptr[i] = stm_val
            
            # Copy to output arrays
            for j in range(n_white):
                w_idx_ptr[w_total + j] = temp_white[j]
            for j in range(n_black):
                b_idx_ptr[b_total + j] = temp_black[j]
            
            w_total += n_white
            b_total += n_black
    
    # Trim to actual size
    white_indices = white_indices[:w_total]
    black_indices = black_indices[:b_total]
    
    return white_indices, white_offsets, black_indices, black_offsets, stm
