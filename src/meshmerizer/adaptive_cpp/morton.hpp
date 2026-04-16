/**
 * @file morton.hpp
 * @brief Morton-key helpers for adaptive octree indexing.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_MORTON_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_MORTON_HPP_

#include <cstdint>

/**
 * @brief Expand 21 low bits so they occupy every third output bit.
 *
 * @param value Unsigned integer coordinate component.
 * @return Bit-interleaved component ready for Morton encoding.
 */
inline std::uint64_t expand_bits_3d(std::uint32_t value) {
    std::uint64_t x = value & 0x1fffffU;
    x = (x | (x << 32U)) & 0x1f00000000ffffULL;
    x = (x | (x << 16U)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8U)) & 0x100f00f00f00f00fULL;
    x = (x | (x << 4U)) & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2U)) & 0x1249249249249249ULL;
    return x;
}

/**
 * @brief Compact every third bit back into a 21-bit coordinate component.
 *
 * @param value Interleaved Morton component.
 * @return Decoded unsigned integer coordinate component.
 */
inline std::uint32_t compact_bits_3d(std::uint64_t value) {
    std::uint64_t x = value & 0x1249249249249249ULL;
    x = (x ^ (x >> 2U)) & 0x10c30c30c30c30c3ULL;
    x = (x ^ (x >> 4U)) & 0x100f00f00f00f00fULL;
    x = (x ^ (x >> 8U)) & 0x1f0000ff0000ffULL;
    x = (x ^ (x >> 16U)) & 0x1f00000000ffffULL;
    x = (x ^ (x >> 32U)) & 0x1fffffULL;
    return static_cast<std::uint32_t>(x);
}

/**
 * @brief Encode three grid coordinates into one 64-bit Morton key.
 *
 * @param x X-axis coordinate component.
 * @param y Y-axis coordinate component.
 * @param z Z-axis coordinate component.
 * @return Bit-interleaved 3-D Morton key.
 */
inline std::uint64_t morton_encode_3d(std::uint32_t x, std::uint32_t y,
                                      std::uint32_t z) {
    return expand_bits_3d(x) | (expand_bits_3d(y) << 1U) |
           (expand_bits_3d(z) << 2U);
}

/**
 * @brief Decode one 64-bit Morton key into three grid coordinates.
 *
 * @param key Bit-interleaved Morton key.
 * @param x Output x-coordinate.
 * @param y Output y-coordinate.
 * @param z Output z-coordinate.
 */
inline void morton_decode_3d(std::uint64_t key, std::uint32_t &x,
                             std::uint32_t &y, std::uint32_t &z) {
    x = compact_bits_3d(key);
    y = compact_bits_3d(key >> 1U);
    z = compact_bits_3d(key >> 2U);
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_MORTON_HPP_
