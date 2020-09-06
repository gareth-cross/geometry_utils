#include "se3.hpp"
#include "numerical_derivative.hpp"
#include "test_utils.hpp"

namespace math {

// Simple test of exponential map by series comparison + numerical derivative.
class TestExponentialMap : public ::testing::Test {
 public:
  template <typename Scalar>
  void TestCase(const Vector<Scalar, 6>& w_u, const Scalar matrix_tol) const {
    // execute the hat operator to get a 4x4 matrix.
    const Matrix<Scalar, 4, 4> w_u_hat = (Matrix<Scalar, 4, 4>() << Skew3(w_u.template head<3>()),
                                          w_u.template tail<3>(), 0, 0, 0, 1)
                                             .finished();
    // compute exponential map power series to 50 terms
    const Matrix<Scalar, 4, 4> T_expected = ExpMatrixSeries(w_u_hat, 50);

    // compute in closed form
    const SE3<Scalar> T_actual = math::SE3Exp(w_u);
    ASSERT_EIGEN_NEAR(T_expected.block(0, 0, 3, 3), T_actual.R.matrix(), matrix_tol);
    ASSERT_EIGEN_NEAR(T_expected.block(0, 3, 3, 1), T_actual.t, matrix_tol);
  }

  void TestMap() const {
    for (const Vector<double, 3> w_vec : Grid3D(-M_PI, M_PI, 0.3)) {
      for (const Vector<double, 3> u_vec : Grid3D(-1.0, 1.0, 0.1)) {
        const Vector<double, 6> xi = (Vector<double, 6>() << w_vec, u_vec).finished();
        TestCase<double>(xi, tol::kPico);
        TestCase<float>(xi.cast<float>(), tol::kMicro);
      }
    }
  }

  //  void TestMapNearZero() const {
  //    TestOmega<double>({1.0e-7, 0.5e-6, 3.5e-8}, tol::kNano, tol::kMicro);
  //    TestOmega<double>({0.0, 0.0, 0.0}, tol::kNano, tol::kMicro);
  //    TestOmega<float>({1.0e-7, 0.5e-6, 3.5e-8}, tol::kNano, tol::kMicro);
  //    TestOmega<float>({0.0, 0.0, 0.0}, tol::kNano, tol::kMicro);
  //  }
};

TEST_FIXTURE(TestExponentialMap, TestMap);

}  // namespace math
