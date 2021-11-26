// Copyright 2020 Gareth Cross
#include "geometry_utils/angle_utils.hpp"

#include "test_utils.hpp"

namespace math {

TEST(AngleUtilsTest, TestModPi) {
  // Test multiples of 2-pi.
  for (int i = -3; i <= 3; ++i) {
    const auto offset = 2 * M_PI * i;
    for (auto angle : {0., M_PI / 6, M_PI / 4, M_PI / 2}) {
      ASSERT_NEAR(angle, ModPi(angle + offset), tol::kNano);
    }
    ASSERT_NEAR(M_PI, ModPi(M_PI + offset), tol::kNano);
    ASSERT_NEAR(M_PI, ModPi(-M_PI + offset), tol::kNano);
  }
}

// Test computing the difference between two angles.
TEST(AngleUtilsTest, TestComputeAngleDelta) {
  // Test multiples of 2-pi.
  for (int i = -3; i <= 3; ++i) {
    const auto offset = 2 * M_PI * i;
    ASSERT_NEAR(0, ComputeAngleDelta(0., 0. + offset), tol::kNano);
    ASSERT_NEAR(0, ComputeAngleDelta(2 * M_PI, 0. + offset), tol::kNano);
    ASSERT_NEAR(0, ComputeAngleDelta(M_PI, -M_PI + offset), tol::kNano);
    ASSERT_NEAR(0, ComputeAngleDelta(-M_PI, M_PI + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 6, ComputeAngleDelta(M_PI / 6, M_PI / 3 + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 3, ComputeAngleDelta(-M_PI / 6, M_PI / 6 + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 2, ComputeAngleDelta(M_PI / 4, 3 * M_PI / 4 + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 2, ComputeAngleDelta(3 * M_PI / 4, 5 * M_PI / 4 + offset), tol::kNano);
    ASSERT_NEAR(-M_PI / 2, ComputeAngleDelta(5 * M_PI / 4, 3 * M_PI / 4 + offset), tol::kNano);
    ASSERT_NEAR(M_PI / 3, ComputeAngleDelta(5 * M_PI / 6 - offset, 7 * M_PI / 6 + offset),
                tol::kNano);
  }
}

}  // namespace math
