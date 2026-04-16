# Critical Code Review Prompt

You are to perform a critical review of the adaptive meshing implementation in the `meshmerizer` repository. Your task is to:

## 1. Review the Plan Document
Read `/Users/willroper/Miscellaneous/meshmerizer/adaptive-plan.md` and understand:
- The overall design goals and architecture
- The current phase completion status
- Any open design questions
- The immediate next steps

## 2. Review the Implementation
Examine the following files for correctness, documentation, and implementation issues:

### Core C++ Headers (in `src/meshmerizer/adaptive_cpp/`):
- `vector3d.hpp` - Vector3d struct
- `bounding_box.hpp` - BoundingBox type
- `particle.hpp` - Particle struct
- `morton.hpp` - Morton key utilities
- `kernel_wendland_c2.hpp` - Wendland C2 kernel
- `particle_grid.hpp` - Top-level particle binning
- `octree_cell.hpp` - Octree cell type and refinement

### Python Bindings:
- `src/meshmerizer/_adaptive.cpp` - C++ extension bridge
- `src/meshmerizer/adaptive_core.py` - Python wrapper

### Tests:
- `tests/test_adaptive_core.py` - Unit tests

## 3. Review Criteria

### Correctness:
- Are algorithms implemented correctly?
- Are there edge cases that could cause bugs (empty inputs, boundary conditions)?
- Is memory management sound?
- Are there potential segmentation faults or undefined behavior?

### Documentation:
- Do all C++ functions/classes have Doxygen comments?
- Do all Python functions have Google-style docstrings?
- Are complex code blocks explained with comments?
- Is naming consistent and descriptive?

### Implementation:
- Are there any obvious performance issues?
- Is the code parallelizable where needed (OpenMP-ready)?
- Is the design consistent with the plan document?

## 4. Testing Requirements
Run the test suite to verify everything works:
```bash
pytest tests/test_adaptive_core.py -v
ruff check .
ruff format .
```

## 5. Fix Issues
If you find any issues, fix them. This may involve:
- Correcting implementation bugs
- Adding missing documentation
- Adding new tests for edge cases

## 6. Update the Plan
After fixing issues, update `adaptive-plan.md`:
- Mark completed phases
- Document any design decisions made
- Note any new issues discovered

## 7. Proceed with Implementation
Once the review is complete and issues are fixed, proceed with the next phase in the plan. The current phase to complete is **Phase 6: Octree Balancing**.

Begin by reading the plan document and understanding the current state of the implementation.
