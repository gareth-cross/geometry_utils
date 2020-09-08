### geometry_utils

Various C++ functions for computing interesting derivatives/jacobians of SO(3).
Wrote these while doing the derivations on paper for my own fun and understanding.
They are unit tested thoroughly, and may prove useful to others as a reference.
I should note that I have only done minimal profiling as a curiosity in a couple of places.
While I tend to avoid any egregious inefficiency, there are still possible optimizations.

- Depends on Eigen for vector/matrix functionality.
- Depends on gtest for running tests.

The functions are header-only, so only tests need building:
```bash
mkdir build
cd build
cmake ..
make
ctest
```
