/**
 * @file poisson_mc.hpp
 * @brief Header-only Marching Cubes isosurface extraction for the screened
 *        Poisson surface reconstruction pipeline (Phase 20e).
 *
 * This header provides a classic Lorensen & Cline Marching Cubes
 * (MC) extractor operating over an adaptive leaf-cell octree.
 *
 * The screened Poisson solver (Kazhdan SGP06; Kazhdan & Hoppe ToG13)
 * produces the coefficient vector @p solution for an indicator function
 * \f$\chi\f$ expressed in a degree-2 (quadratic) B-spline basis:
 *
 * \f[ \chi(p) = \sum_j x_j B_j(p). \f]
 *
 * We evaluate \f$\chi\f$ at octree leaf-cell corners, choose an isovalue
 * (default: mean of \f$\chi\f$ at oriented samples), and extract the
 * corresponding isosurface with Marching Cubes.
 *
 * @par References
 * - Lorensen, W. E. & Cline, H. E. "Marching Cubes: A High Resolution 3D
 *   Surface Construction Algorithm", *Computer Graphics (SIGGRAPH)* 21(4),
 *   163--169 (1987).
 * - Kazhdan, M., Bolitho, M. & Hoppe, H. "Poisson Surface Reconstruction",
 *   *Proc. SGP* (2006).
 * - Kazhdan, M. & Hoppe, H. "Screened Poisson Surface Reconstruction",
 *   *ACM Trans. Graph.* 32(3), Art. 29 (2013).
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_POISSON_MC_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_POISSON_MC_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "bounding_box.hpp"
#include "octree_cell.hpp"
#include "poisson_basis.hpp"
#include "poisson_rhs.hpp"  // find_overlapping_dofs
#include "progress_bar.hpp"
#include "vector3d.hpp"

/* =======================================================================
 * Classic Marching Cubes tables (Lorensen & Cline 1987)
 *
 * Corner numbering (standard):
 *   0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0)
 *   4: (0,0,1), 5: (1,0,1), 6: (1,1,1), 7: (0,1,1)
 *
 * Edge numbering (standard):
 *   0: 0-1,  1: 1-2,  2: 2-3,  3: 3-0,
 *   4: 4-5,  5: 5-6,  6: 6-7,  7: 7-4,
 *   8: 0-4,  9: 1-5, 10: 2-6, 11: 3-7.
 * ======================================================================= */

/**
 * @brief Marching Cubes edge intersection mask table.
 *
 * For each of the 256 cases, this table provides a 12-bit mask where bit
 * e is set when edge e is intersected by the isosurface.
 */
static constexpr std::uint16_t MC_EDGE_TABLE[256] = {
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905,
    0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x099, 0x393, 0x29a,
    0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93,
    0xf99, 0xe90, 0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9,
    0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6,
    0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f,
    0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5,
    0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a,
    0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
    0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9,
    0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6,
    0x4ca, 0x5c3, 0x6c9, 0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f,
    0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
    0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a,
    0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663,
    0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 0xd30, 0xc39,
    0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
    0x13a, 0x033, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f,
    0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605,
    0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
};

/**
 * @brief Marching Cubes triangle table.
 *
 * For each of the 256 cases, lists the edge indices (0-11) that form
 * the triangle vertices. The list is a sequence of triples, terminated by
 * -1. The maximum number of triangle vertices per case is 15.
 */
static constexpr std::int8_t MC_TRI_TABLE[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
};

/* =======================================================================
 * Helper utilities
 * ======================================================================= */

/**
 * @brief Corner offsets in (x,y,z) as bits for the standard numbering.
 */
inline void mc_corner_offset_bits(std::size_t corner_index,
                                  std::uint32_t &ox,
                                  std::uint32_t &oy,
                                  std::uint32_t &oz) {
    ox = (corner_index & 1U) ? 1U : 0U;
    oy = (corner_index & 2U) ? 1U : 0U;
    oz = (corner_index & 4U) ? 1U : 0U;
}

/**
 * @brief Return the two corners incident to a given MC edge.
 */
inline void mc_edge_corners(std::size_t edge_index,
                            std::uint8_t &c0,
                            std::uint8_t &c1) {
    static constexpr std::uint8_t EDGE_CORNERS[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
        {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    c0 = EDGE_CORNERS[edge_index][0];
    c1 = EDGE_CORNERS[edge_index][1];
}

/**
 * @brief Linearly interpolate an isosurface vertex on an edge.
 *
 * @param p0 Edge endpoint 0 (world space).
 * @param p1 Edge endpoint 1 (world space).
 * @param v0 Scalar value at @p p0.
 * @param v1 Scalar value at @p p1.
 * @param isovalue Target isosurface value.
 * @return Interpolated point in world space.
 */
inline Vector3d mc_interpolate_vertex(const Vector3d &p0,
                                      const Vector3d &p1,
                                      double v0,
                                      double v1,
                                      double isovalue) {
    const double dv = v1 - v0;
    double t = 0.5;
    if (dv != 0.0) {
        t = (isovalue - v0) / dv;
        if (t < 0.0) {
            t = 0.0;
        } else if (t > 1.0) {
            t = 1.0;
        }
    }
    return {
        p0.x + t * (p1.x - p0.x),
        p0.y + t * (p1.y - p0.y),
        p0.z + t * (p1.z - p0.z),
    };
}

/**
 * @brief Collision-free ordered edge key for the MC edge cache.
 *
 * Previous implementation used a multiplicative hash (lo * PRIME + hi)
 * which could produce collisions between distinct edge pairs, causing
 * incorrect vertex sharing and non-manifold edges.  Using an ordered
 * std::pair as the map key is collision-free by construction.
 */
using MCEdgeKey = std::pair<std::uint64_t, std::uint64_t>;

/**
 * @brief Create an ordered edge key from two packed grid corners.
 */
inline MCEdgeKey mc_edge_key(std::uint64_t packed_a,
                             std::uint64_t packed_b) {
    if (packed_a <= packed_b) {
        return {packed_a, packed_b};
    }
    return {packed_b, packed_a};
}

/**
 * @brief Hash functor for MCEdgeKey (pair of uint64_t).
 *
 * Combines both halves with a mixing constant to spread bits.
 */
struct MCEdgeKeyHash {
    std::size_t operator()(const MCEdgeKey &k) const noexcept {
        static constexpr std::uint64_t PRIME =
            0x9e3779b97f4a7c15ULL;
        return static_cast<std::size_t>(
            k.first * PRIME ^ k.second);
    }
};

/* =======================================================================
 * Virtual boundary cell for closing the adaptive MC mesh.
 *
 * When DOFs are restricted to max_depth, the max_depth leaves form an
 * incomplete region surrounded by coarser cells.  Running MC only on
 * max_depth leaves produces a mesh with open boundary faces where the
 * fine region borders coarser cells.
 *
 * To close the mesh, we create "virtual" max_depth-sized cells just
 * outside each boundary face.  These virtual cells have chi evaluated
 * at their corners via the B-spline field (which decays smoothly to ~0
 * outside the DOF region), so MC can properly generate the closing
 * surface.  This adds O(N^{2/3}) cells -- a thin shell, not a full
 * uniform grid.
 *
 * Cf. Kazhdan & Hoppe (ToG13) FEMTree.LevelSet.3D.inl, where corner
 * values are propagated from fine leaves to coarse ancestors, and
 * coarse faces accumulate iso-edge fragments from fine cells.  Our
 * approach achieves the same effect more simply by materializing
 * virtual cells at the boundary.
 * ======================================================================= */

/**
 * @brief A virtual max_depth-sized cell created at the boundary of
 *        the fine-leaf region to close the MC mesh.
 */
struct VirtualCell {
    BoundingBox bounds;
    std::array<double, 8> corner_values;
    /** Fine-grid min-corner coordinates (for edge key generation). */
    std::uint32_t gx0, gy0, gz0;
};

/* =======================================================================
 * Public API
 * ======================================================================= */

/**
 * @brief Evaluate chi at a single world-space corner position.
 *
 * Uses the B-spline field chi(p) = sum_j x_j B_j(p) with DOF lookup
 * via find_overlapping_dofs().  Results are cached in @p corner_cache
 * so each unique grid corner is evaluated exactly once.
 *
 * @param corner World-space corner position.
 * @param packed Packed fine-grid coordinate key.
 * @param solution Solution vector.
 * @param cells Octree cells.
 * @param cell_to_dof Cell-to-DOF mapping.
 * @param hash Poisson leaf hash.
 * @param base_resolution Top-level grid resolution.
 * @param corner_cache Global cache (packed coord -> chi value).
 * @param dof_indices Scratch buffer (reused across calls).
 * @param weights Scratch buffer (reused across calls).
 * @return Chi value at the corner.
 */
inline double evaluate_chi_at_point(
    const Vector3d &corner,
    std::uint64_t packed,
    const std::vector<double> &solution,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const PoissonLeafHash &hash,
    std::uint32_t base_resolution,
    std::unordered_map<std::uint64_t, double> &corner_cache,
    std::vector<std::int64_t> &dof_indices,
    std::vector<double> &weights) {
    auto it = corner_cache.find(packed);
    if (it != corner_cache.end()) {
        return it->second;
    }

    find_overlapping_dofs(corner, hash, cells, cell_to_dof,
                          dof_indices, weights, base_resolution);

    double chi = 0.0;
    for (std::size_t k = 0; k < dof_indices.size(); ++k) {
        const std::int64_t dof = dof_indices[k];
        if (dof < 0) {
            continue;
        }
        const std::size_t idx = static_cast<std::size_t>(dof);
        if (idx >= solution.size()) {
            continue;
        }
        chi += solution[idx] * weights[k];
    }
    corner_cache.emplace(packed, chi);
    return chi;
}

/**
 * @brief Evaluate the Poisson indicator function chi at the eight
 *        corners of each max_depth leaf cell, then create virtual
 *        boundary cells to close the mesh.
 *
 * For each max_depth leaf, we evaluate chi(p) = sum_j x_j B_j(p) at
 * the 8 cube corners.  A global corner cache ensures shared corners
 * produce identical values.
 *
 * After evaluating real leaves, we identify boundary faces -- faces
 * of max_depth leaves where the neighbor is a coarser cell (or domain
 * boundary).  For each such face we create a virtual max_depth cell
 * on the outside, evaluate chi at its corners, and append it to the
 * output.  This closes the MC mesh without allocating a full uniform
 * grid (Kazhdan & Hoppe ToG13 section 4).
 *
 * @param solution Solution vector x (size = n_dofs).
 * @param cells Full octree cell array.
 * @param cell_to_dof Mapping from cell index to DOF index.
 * @param hash Spatial hash of leaf cells.
 * @param base_resolution Top-level cells per axis.
 * @param max_depth Maximum octree depth.
 * @param n_cells Number of leaf cells (progress reporting).
 * @param domain Domain bounding box.
 * @param corner_values Output per-cell corner values.
 * @param virtual_cells Output virtual boundary cells.
 */
inline void evaluate_chi_at_corners(
    const std::vector<double> &solution,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const PoissonLeafHash &hash,
    std::uint32_t base_resolution,
    std::uint32_t max_depth,
    std::size_t n_cells,
    const BoundingBox &domain,
    std::vector<std::array<double, 8>> &corner_values,
    std::vector<VirtualCell> &virtual_cells) {
    corner_values.assign(cells.size(), std::array<double, 8>{});
    virtual_cells.clear();

    ProgressBar progress("Evaluating chi corners",
                         n_cells > 0 ? n_cells : 1);

    // Global corner cache: packed fine-grid coordinate -> chi value.
    std::unordered_map<std::uint64_t, double> corner_cache;
    corner_cache.reserve(n_cells * 4);

    std::vector<std::int64_t> dof_indices;
    std::vector<double> weights;
    dof_indices.reserve(27);
    weights.reserve(27);

    // Compute fine-grid cell width in world space.
    const double fine_cells_per_axis =
        static_cast<double>(base_resolution) *
        static_cast<double>(1U << max_depth);
    const double cell_width_x =
        (domain.max.x - domain.min.x) / fine_cells_per_axis;
    const double cell_width_y =
        (domain.max.y - domain.min.y) / fine_cells_per_axis;
    const double cell_width_z =
        (domain.max.z - domain.min.z) / fine_cells_per_axis;

    // Max fine-grid coordinate.
    const std::uint32_t max_fine_coord =
        base_resolution * (1U << max_depth);

    // Collect max_depth leaf fine-grid positions for boundary
    // detection (set of occupied fine-grid positions).
    std::unordered_map<std::uint64_t, std::size_t> fine_leaf_map;
    fine_leaf_map.reserve(n_cells);

    // Pass 1: Evaluate chi at max_depth leaf corners.
    for (std::size_t ci = 0; ci < cells.size(); ++ci) {
        if (!cells[ci].is_leaf) {
            continue;
        }
        if (cells[ci].depth != max_depth) {
            continue;
        }
        const OctreeCell &cell = cells[ci];

        std::uint32_t cell_x = 0U, cell_y = 0U, cell_z = 0U;
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
        // At max_depth, span = 1.
        const std::uint32_t gx0 = cell_x;
        const std::uint32_t gy0 = cell_y;
        const std::uint32_t gz0 = cell_z;

        // Register in fine_leaf_map for boundary detection.
        fine_leaf_map[PoissonLeafHash::pack(gx0, gy0, gz0)] = ci;

        std::array<double, 8> values;
        for (std::size_t corner_index = 0; corner_index < 8;
             ++corner_index) {
            std::uint32_t ox, oy, oz;
            mc_corner_offset_bits(corner_index, ox, oy, oz);
            const std::uint32_t gx = gx0 + ox;
            const std::uint32_t gy = gy0 + oy;
            const std::uint32_t gz = gz0 + oz;
            const std::uint64_t packed =
                PoissonLeafHash::pack(gx, gy, gz);

            const Vector3d corner = {
                (corner_index & 1U) ? cell.bounds.max.x
                                    : cell.bounds.min.x,
                (corner_index & 2U) ? cell.bounds.max.y
                                    : cell.bounds.min.y,
                (corner_index & 4U) ? cell.bounds.max.z
                                    : cell.bounds.min.z,
            };

            values[corner_index] = evaluate_chi_at_point(
                corner, packed, solution, cells, cell_to_dof,
                hash, base_resolution, corner_cache,
                dof_indices, weights);
        }

        corner_values[ci] = values;
        progress.tick();
    }
    progress.finish();

    // Pass 2: Create virtual boundary cells.
    // For each max_depth leaf, check 6 face neighbors.  If the
    // neighbor position is NOT occupied by another max_depth leaf,
    // create a virtual cell there.
    //
    // Direction offsets: +X, -X, +Y, -Y, +Z, -Z
    static const std::int32_t DIR_DX[6] = {1, -1, 0, 0, 0, 0};
    static const std::int32_t DIR_DY[6] = {0, 0, 1, -1, 0, 0};
    static const std::int32_t DIR_DZ[6] = {0, 0, 0, 0, 1, -1};

    // Track which virtual cells we've already created to avoid
    // duplicates (keyed by packed fine-grid min-corner).
    std::unordered_map<std::uint64_t, std::size_t>
        virtual_cell_map;
    virtual_cell_map.reserve(n_cells / 4);  // Rough estimate.

    ProgressBar virt_progress(
        "Creating virtual boundary cells", n_cells > 0 ? n_cells : 1);

    for (const auto &entry : fine_leaf_map) {
        const std::size_t ci = entry.second;
        const OctreeCell &cell = cells[ci];

        std::uint32_t cell_x = 0U, cell_y = 0U, cell_z = 0U;
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);

        for (int dir = 0; dir < 6; ++dir) {
            const std::int32_t nx =
                static_cast<std::int32_t>(cell_x) + DIR_DX[dir];
            const std::int32_t ny =
                static_cast<std::int32_t>(cell_y) + DIR_DY[dir];
            const std::int32_t nz =
                static_cast<std::int32_t>(cell_z) + DIR_DZ[dir];

            // Skip if out of domain bounds.
            if (nx < 0 || ny < 0 || nz < 0) {
                continue;
            }
            if (static_cast<std::uint32_t>(nx) >= max_fine_coord ||
                static_cast<std::uint32_t>(ny) >= max_fine_coord ||
                static_cast<std::uint32_t>(nz) >= max_fine_coord) {
                continue;
            }

            const std::uint32_t ngx =
                static_cast<std::uint32_t>(nx);
            const std::uint32_t ngy =
                static_cast<std::uint32_t>(ny);
            const std::uint32_t ngz =
                static_cast<std::uint32_t>(nz);
            const std::uint64_t npacked =
                PoissonLeafHash::pack(ngx, ngy, ngz);

            // Skip if a real max_depth leaf exists there.
            if (fine_leaf_map.count(npacked)) {
                continue;
            }

            // Skip if we already created a virtual cell there.
            if (virtual_cell_map.count(npacked)) {
                continue;
            }

            // Create a virtual cell at this position.
            VirtualCell vc;
            vc.gx0 = ngx;
            vc.gy0 = ngy;
            vc.gz0 = ngz;

            // Compute world-space bounds.
            vc.bounds.min = {
                domain.min.x +
                    static_cast<double>(ngx) * cell_width_x,
                domain.min.y +
                    static_cast<double>(ngy) * cell_width_y,
                domain.min.z +
                    static_cast<double>(ngz) * cell_width_z,
            };
            vc.bounds.max = {
                vc.bounds.min.x + cell_width_x,
                vc.bounds.min.y + cell_width_y,
                vc.bounds.min.z + cell_width_z,
            };

            // Evaluate chi at the 8 corners.
            for (std::size_t corner_index = 0;
                 corner_index < 8; ++corner_index) {
                std::uint32_t ox, oy, oz;
                mc_corner_offset_bits(corner_index, ox, oy, oz);
                const std::uint32_t gx = ngx + ox;
                const std::uint32_t gy = ngy + oy;
                const std::uint32_t gz = ngz + oz;
                const std::uint64_t cpacked =
                    PoissonLeafHash::pack(gx, gy, gz);

                const Vector3d corner = {
                    (corner_index & 1U) ? vc.bounds.max.x
                                        : vc.bounds.min.x,
                    (corner_index & 2U) ? vc.bounds.max.y
                                        : vc.bounds.min.y,
                    (corner_index & 4U) ? vc.bounds.max.z
                                        : vc.bounds.min.z,
                };

                vc.corner_values[corner_index] =
                    evaluate_chi_at_point(
                        corner, cpacked, solution, cells,
                        cell_to_dof, hash, base_resolution,
                        corner_cache, dof_indices, weights);
            }

            virtual_cell_map[npacked] = virtual_cells.size();
            virtual_cells.push_back(vc);
        }

        virt_progress.tick();
    }
    virt_progress.finish();
}

/**
 * @brief Compute a default isovalue as the mean \f$\chi\f$ value at sample
 *        positions.
 *
 * This follows the common practice in Poisson reconstruction pipelines:
 * the implicit surface is extracted at a level set representative of the
 * sample distribution. Here we choose the arithmetic mean over samples.
 *
 * @param positions Pointer to sample positions.
 * @param n_samples Number of samples.
 * @param solution Solution vector x.
 * @param cells Full octree cell array.
 * @param cell_to_dof Cell-to-DOF mapping.
 * @param hash Spatial hash of leaf cells.
 * @param base_resolution Top-level cells per axis.
 * @return Mean isovalue.
 */
inline double compute_isovalue(
    const Vector3d *positions,
    std::size_t n_samples,
    const std::vector<double> &solution,
    const std::vector<OctreeCell> &cells,
    const std::vector<std::int64_t> &cell_to_dof,
    const PoissonLeafHash &hash,
    std::uint32_t base_resolution) {
    if (positions == nullptr || n_samples == 0) {
        return 0.0;
    }

    ProgressBar progress("Computing isovalue", n_samples);

    std::vector<std::int64_t> dof_indices;
    std::vector<double> weights;
    dof_indices.reserve(27);
    weights.reserve(27);

    double accum = 0.0;
    for (std::size_t s = 0; s < n_samples; ++s) {
        find_overlapping_dofs(positions[s], hash, cells, cell_to_dof,
                              dof_indices, weights, base_resolution);
        double chi = 0.0;
        for (std::size_t k = 0; k < dof_indices.size(); ++k) {
            const std::int64_t dof = dof_indices[k];
            if (dof < 0) {
                continue;
            }
            const std::size_t idx = static_cast<std::size_t>(dof);
            if (idx >= solution.size()) {
                continue;
            }
            chi += solution[idx] * weights[k];
        }
        accum += chi;
        progress.tick();
    }
    progress.finish();
    return accum / static_cast<double>(n_samples);
}

/**
 * @brief Run Marching Cubes on a single cell given its bounds, corner
 *        values, and fine-grid min-corner coordinates.
 *
 * This is a shared helper used for both real max_depth leaves and
 * virtual boundary cells.
 *
 * @param bounds Cell bounding box.
 * @param vals Corner chi values.
 * @param gx0 Fine-grid min-corner X.
 * @param gy0 Fine-grid min-corner Y.
 * @param gz0 Fine-grid min-corner Z.
 * @param isovalue Isosurface level.
 * @param edge_cache Shared edge vertex cache.
 * @param vertices Output vertex positions.
 * @param triangles Output triangle index triplets.
 */
inline void mc_process_cell(
    const BoundingBox &bounds,
    const std::array<double, 8> &vals,
    std::uint32_t gx0,
    std::uint32_t gy0,
    std::uint32_t gz0,
    double isovalue,
    std::unordered_map<MCEdgeKey, std::uint32_t,
                       MCEdgeKeyHash> &edge_cache,
    std::vector<Vector3d> &vertices,
    std::vector<std::array<std::uint32_t, 3>> &triangles) {
    // Compute cube case index.
    std::uint8_t cube_index = 0U;
    for (std::size_t corner_index = 0; corner_index < 8;
         ++corner_index) {
        if (vals[corner_index] > isovalue) {
            cube_index |=
                static_cast<std::uint8_t>(1U << corner_index);
        }
    }

    const std::uint16_t edge_mask = MC_EDGE_TABLE[cube_index];
    if (edge_mask == 0U) {
        return;
    }

    // Corner positions in world space.
    Vector3d corner_pos[8];
    for (std::size_t ci = 0; ci < 8; ++ci) {
        corner_pos[ci] = {
            (ci & 1U) ? bounds.max.x : bounds.min.x,
            (ci & 2U) ? bounds.max.y : bounds.min.y,
            (ci & 4U) ? bounds.max.z : bounds.min.z,
        };
    }

    // Fine-grid packed coordinates for the 8 corners.
    // At max_depth, span = 1.
    std::uint64_t packed_corner[8];
    for (std::size_t ci = 0; ci < 8; ++ci) {
        std::uint32_t ox, oy, oz;
        mc_corner_offset_bits(ci, ox, oy, oz);
        packed_corner[ci] = PoissonLeafHash::pack(
            gx0 + ox, gy0 + oy, gz0 + oz);
    }

    // For each intersected edge, create/reuse the vertex.
    std::uint32_t edge_vertex_index[12] = {0};
    for (std::size_t e = 0; e < 12; ++e) {
        if ((edge_mask & (1U << e)) == 0U) {
            continue;
        }

        std::uint8_t c0, c1;
        mc_edge_corners(e, c0, c1);
        const MCEdgeKey key =
            mc_edge_key(packed_corner[c0], packed_corner[c1]);

        auto it = edge_cache.find(key);
        if (it != edge_cache.end()) {
            edge_vertex_index[e] = it->second;
            continue;
        }

        const Vector3d p = mc_interpolate_vertex(
            corner_pos[c0], corner_pos[c1],
            vals[c0], vals[c1], isovalue);

        const std::uint32_t new_index =
            static_cast<std::uint32_t>(vertices.size());
        vertices.push_back(p);
        edge_cache.emplace(key, new_index);
        edge_vertex_index[e] = new_index;
    }

    // Emit triangles.
    for (std::size_t t = 0; t < 16; t += 3) {
        const std::int8_t e0 = MC_TRI_TABLE[cube_index][t];
        if (e0 < 0) {
            break;
        }
        const std::int8_t e1 = MC_TRI_TABLE[cube_index][t + 1];
        const std::int8_t e2 = MC_TRI_TABLE[cube_index][t + 2];
        triangles.push_back({
            edge_vertex_index[static_cast<std::size_t>(e0)],
            edge_vertex_index[static_cast<std::size_t>(e1)],
            edge_vertex_index[static_cast<std::size_t>(e2)],
        });
    }
}

/**
 * @brief Extract an isosurface using Marching Cubes over max_depth
 *        leaf cells plus virtual boundary cells.
 *
 * Runs classic MC (Lorensen & Cline 1987) on:
 * 1. All real max_depth leaf cells (from @p corner_values).
 * 2. Virtual boundary cells (from @p virtual_cells) that close the
 *    mesh at the boundary of the fine-leaf region.
 *
 * Edge vertices are shared between cells via a collision-free edge
 * cache keyed by fine-grid corner coordinates.
 *
 * @param cells Full octree cell array.
 * @param corner_values Per-cell corner samples (aligned with cells).
 * @param virtual_cells Virtual boundary cells from evaluate_chi.
 * @param isovalue Surface level.
 * @param domain Global domain bounding box.
 * @param base_resolution Number of top-level cells per axis.
 * @param max_depth Maximum octree depth.
 * @param vertices Output vertex positions.
 * @param triangles Output triangle index triplets.
 */
inline void extract_isosurface(
    const std::vector<OctreeCell> &cells,
    const std::vector<std::array<double, 8>> &corner_values,
    const std::vector<VirtualCell> &virtual_cells,
    double isovalue,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t max_depth,
    std::vector<Vector3d> &vertices,
    std::vector<std::array<std::uint32_t, 3>> &triangles) {
    (void)domain;
    (void)base_resolution;

    vertices.clear();
    triangles.clear();

    std::size_t n_leaves = 0;
    for (const auto &c : cells) {
        if (c.is_leaf && c.depth == max_depth) {
            ++n_leaves;
        }
    }

    const std::size_t total =
        n_leaves + virtual_cells.size();
    ProgressBar progress(
        "Marching Cubes", total > 0 ? total : 1);

    std::unordered_map<MCEdgeKey, std::uint32_t,
                       MCEdgeKeyHash> edge_cache;
    edge_cache.reserve(total * 6);

    // Pass 1: Real max_depth leaf cells.
    for (std::size_t ci = 0; ci < cells.size(); ++ci) {
        if (!cells[ci].is_leaf) {
            continue;
        }
        if (cells[ci].depth != max_depth) {
            continue;
        }
        if (ci >= corner_values.size()) {
            progress.tick();
            continue;
        }

        const OctreeCell &cell = cells[ci];

        std::uint32_t cell_x = 0U, cell_y = 0U, cell_z = 0U;
        morton_decode_3d(cell.morton_key, cell_x, cell_y,
                         cell_z);
        // At max_depth, span = 1.

        mc_process_cell(
            cell.bounds, corner_values[ci],
            cell_x, cell_y, cell_z,
            isovalue, edge_cache, vertices, triangles);

        progress.tick();
    }

    // Pass 2: Virtual boundary cells.
    for (const auto &vc : virtual_cells) {
        mc_process_cell(
            vc.bounds, vc.corner_values,
            vc.gx0, vc.gy0, vc.gz0,
            isovalue, edge_cache, vertices, triangles);

        progress.tick();
    }

    progress.finish();
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_POISSON_MC_HPP_
