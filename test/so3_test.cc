#include <chrono>
#include <iostream>

#include "numerical_derivative.hpp"
#include "so3.hpp"
#include "test_utils.hpp"

// TODO(gareth): I think gtest has some tools for testing both double/float.
// Should probably use those here instead of calling manaually.
namespace math {

// Test skew-symmetric operator.
TEST(SO3Test, TestSkew3) {
  using Vector3d = Vector<double, 3>;
  using Matrix3d = Matrix<double, 3, 3>;
  const Vector3d x = (Vector3d() << 1, 2, 3).finished();
  const Vector3d y = (Vector3d() << 1, 1, 1).finished();
  EXPECT_EIGEN_NEAR(Vector3d::Zero(), Skew3(x) * x, tol::kPico);
  EXPECT_EIGEN_NEAR(x.cross(y), Skew3(x) * y, tol::kPico);
  EXPECT_EIGEN_NEAR(Matrix3d::Zero(), Skew3(x) + Skew3(x).transpose(), tol::kPico);
}

// Test quaternion multiplication.
TEST(SO3Test, TestQuaternionMulMatrix) {
  const auto to_vec = [](const Quaternion<double>& q) -> Vector<double, 4> {
    return Vector<double, 4>(q.w(), q.x(), q.y(), q.z());
  };
  const Quaternion<double> q0{-0.5, 0.2, 0.1, 0.8};
  const Quaternion<double> q1{0.4, -0.3, 0.2, 0.45};
  EXPECT_EIGEN_NEAR(to_vec(q0 * q1), QuaternionMulMatrix(q0) * to_vec(q1), tol::kPico);
  EXPECT_EIGEN_NEAR(to_vec(q1 * q0), QuaternionMulMatrix(q1) * to_vec(q0), tol::kPico);
}

// Simple test of exponential map by series comparison + numerical derivative.
class TestQuaternionExp : public ::testing::Test {
 public:
  template <typename Scalar>
  void TestOmega(const Vector<Scalar, 3>& w, const Scalar matrix_tol,
                 const Scalar deriv_tol) const {
    // check that the derivative and non-derivative versions are the same
    const Quaternion<Scalar> just_q = math::QuaternionExp(w);
    const QuaternionExpDerivative<Scalar> q_and_deriv{w};
    ASSERT_EIGEN_NEAR(just_q.matrix(), q_and_deriv.q.matrix(), tol::kPico);

    // compare to the exponential map as a power series ~ 50 terms
    ASSERT_EIGEN_NEAR(ExpMatrixSeries(Skew3(w), 50), q_and_deriv.q.matrix(), matrix_tol)
        << "w = " << w.transpose();

    // compare to Eigen implementation for good measure
    const Eigen::AngleAxis<Scalar> aa(w.norm(), w.normalized());
    ASSERT_EIGEN_NEAR(aa.toRotationMatrix(), q_and_deriv.q.matrix(), matrix_tol)
        << "w = " << w.transpose();

    // check derivative numerically
    const Matrix<Scalar, 4, 3> J_numerical =
        NumericalJacobian(w, [](const Vector<Scalar, 3>& w) -> Vector<Scalar, 4> {
          // convert to correct order here
          const Quaternion<Scalar> q = math::QuaternionExp(w);
          return Vector<Scalar, 4>(q.w(), q.x(), q.y(), q.z());
        });
    ASSERT_EIGEN_NEAR(J_numerical, q_and_deriv.q_D_w, deriv_tol) << "w = " << w.transpose();
  }

  void Test() const {
    // vec3d is not aligned (not 16 byte multiple)
    // clang-format off
    const std::vector<Eigen::Vector3d> vectors = {
      {-M_PI, 0, 0},
      {0, M_PI, 0},
      {0, 0, -M_PI},
    };
    // clang-format on
    // test near pi
    for (const Eigen::Vector3d& w : vectors) {
      TestOmega<double>(w, tol::kPico, tol::kNano);
      TestOmega<float>(w.cast<float>(), tol::kMicro, tol::kMilli / 10);
    }
    for (const Eigen::Vector3d& w : kRandomRotationVectors) {
      TestOmega<double>(w, tol::kPico, tol::kNano);
      TestOmega<float>(w.cast<float>(), tol::kMicro, tol::kMilli / 10);
    }
  }

  void TestNearZero() const {
    TestOmega<double>({1.0e-7, 0.5e-6, 3.5e-8}, tol::kNano, tol::kMicro);
    TestOmega<double>({0.0, 0.0, 0.0}, tol::kNano, tol::kMicro);
    TestOmega<float>({1.0e-7, 0.5e-6, 3.5e-8}, tol::kNano, tol::kMicro);
    TestOmega<float>({0.0, 0.0, 0.0}, tol::kNano, tol::kMicro);
  }
};

TEST_FIXTURE(TestQuaternionExp, Test)
TEST_FIXTURE(TestQuaternionExp, TestNearZero)

// Check that RotationLog does the inverse of QuaternionExp.
TEST(SO3Test, TestRotationLog) {
  // test quaternion
  const Vector<double, 3> v1{-0.7, 0.23, 0.4};
  const Quaternion<double> r1 = QuaternionExp(v1);
  EXPECT_EIGEN_NEAR(v1, RotationLog(r1), tol::kPico);
  // test matrix
  const Vector<float, 3> v2{0.01, -0.5, 0.03};
  const Quaternion<float> r2 = QuaternionExp(v2);
  EXPECT_EIGEN_NEAR(v2, RotationLog(r2.matrix()), tol::kMicro);
  // test identity
  const auto zero = Vector<double, 3>::Zero();
  EXPECT_EIGEN_NEAR(zero, RotationLog(Quaternion<double>::Identity()), tol::kPico);
  EXPECT_EIGEN_NEAR(zero.cast<float>(), RotationLog(Matrix<float, 3, 3>::Identity()), tol::kPico);
  // test some randomly sampled vectors
  for (const Eigen::Vector3d& w : kRandomRotationVectors) {
    const Eigen::Matrix3d R = QuaternionExp(w).matrix();
    // make sure it's the same rotation we get back out
    EXPECT_EIGEN_NEAR(R, QuaternionExp(RotationLog(R)).matrix(), tol::kNano);
    EXPECT_EIGEN_NEAR(R.cast<float>(), QuaternionExp(RotationLog(R.cast<float>())).matrix(),
                      tol::kMicro);
  }
}

// Test the SO(3) jacobian.
class TestSO3Jacobian : public ::testing::Test {
 public:
  template <typename Scalar>
  static void TestJacobian(const Vector<Scalar, 3>& w_a, const Scalar deriv_tol) {
    const Matrix<Scalar, 3, 3> J_analytical = math::SO3Jacobian(w_a);
    // This jacobian is only valid for small `w`, so evaluate about zero.
    const Matrix<Scalar, 3, 3> J_numerical =
        NumericalJacobian(Vector<Scalar, 3>::Zero(),
                          [&](const Vector<Scalar, 3>& w) { return QuaternionExp(w_a + w); });
    EXPECT_EIGEN_NEAR(J_numerical, J_analytical, deriv_tol);
  }

  void TestGeneral() {
    for (const Eigen::Vector3d& w : kRandomRotationVectors) {
      TestJacobian<double>(w, tol::kNano);
      TestJacobian<float>(w.cast<float>(), tol::kMilli / 10);
    }
  }

  void TestNearZero() {}
};

TEST_FIXTURE(TestSO3Jacobian, TestGeneral)
TEST_FIXTURE(TestSO3Jacobian, TestNearZero)

// Test the derivative of the exponential map, matrix form.
class TestMatrixExpDerivative : public ::testing::Test {
 public:
  template <typename Scalar>
  static Vector<Scalar, 9> VecExpMatrix(const Vector<Scalar, 3>& w) {
    // Convert to vectorized format.
    const Matrix<Scalar, 3, 3> R = math::QuaternionExp(w).matrix();
    return Eigen::Map<const Vector<Scalar, 9>>(R.data());
  }

  template <typename Scalar>
  static void TestDerivative(const Vector<Scalar, 3>& w, const Scalar deriv_tol) {
    const Matrix<Scalar, 9, 3> D_w = math::SO3ExpMatrixDerivative(w);
    const Matrix<Scalar, 9, 3> J_numerical =
        NumericalJacobian(w, &TestMatrixExpDerivative::VecExpMatrix<Scalar>);
    ASSERT_EIGEN_NEAR(J_numerical, D_w, deriv_tol);
  }

  void TestGeneral() {
    for (const Eigen::Vector3d& w : kRandomRotationVectors) {
      TestDerivative<double>(w, tol::kNano / 10);
      TestDerivative<float>(w.cast<float>(), tol::kMilli / 10);
    }
  }

  void TestNearZero() {
    TestDerivative<double>({-1.0e-7, 1.0e-8, 0.5e-6}, tol::kMicro);
    TestDerivative<float>({-1.0e-7, 1.0e-8, 0.5e-6}, tol::kMicro);

    // at exactly zero it should be identically equal to the generators of SO(3)
    const Matrix<double, 9, 3> J_at_zero =
        math::SO3ExpMatrixDerivative(Vector<double, 3>::Zero().eval());
    const auto i_hat = Vector<double, 3>::UnitX();
    const auto j_hat = Vector<double, 3>::UnitY();
    const auto k_hat = Vector<double, 3>::UnitZ();
    EXPECT_EIGEN_NEAR(Skew3(-i_hat), J_at_zero.block(0, 0, 3, 3), tol::kPico);
    EXPECT_EIGEN_NEAR(Skew3(-j_hat), J_at_zero.block(3, 0, 3, 3), tol::kPico);
    EXPECT_EIGEN_NEAR(Skew3(-k_hat), J_at_zero.block(6, 0, 3, 3), tol::kPico);
  }
};

TEST_FIXTURE(TestMatrixExpDerivative, TestGeneral)
TEST_FIXTURE(TestMatrixExpDerivative, TestNearZero)

// Have to be careful when testing this method numerically, since the output of log() can
// jump around if the rotation R * exp(w) is large.
TEST(SO3Test, SO3LogMulExpDerivative) {
  // create the matrix R we multiply against
  const Vector<double, 3> R_log{0.6, -0.1, 0.4};
  const Quaternion<double> R = math::QuaternionExp(R_log);

  // functor that holds R fixed and multiplies on w
  const auto fix_r_functor = [&](const Vector<double, 3>& w) -> Vector<double, 3> {
    return math::RotationLog(R * math::QuaternionExp(w));
  };

  // try a bunch of values for omega
  // clang-format off
  const std::vector<Vector<double, 3>> samples = {
    {0.6, -0.1, 0.4},
    {0.8, 0.0, 0.2},
    {-1.2, 0.6, 1.5},
    {0.0, 0.0, 0.1},
    {1.5, 1.7, -1.2},
    {-0.3, 0.3, 0.3},
    {M_PI / 2, 0, 0},
    {0, 0, -M_PI / 4},
    {0.5, 0.0, M_PI / 2},
  };
  // clang-format on
  for (const auto& w : samples) {
    const Matrix<double, 3, 3> J_analytical = math::SO3LogMulExpDerivative(R, w);
    const Matrix<double, 3, 3> J_numerical = NumericalJacobian(w, fix_r_functor);
    ASSERT_EIGEN_NEAR(J_numerical, J_analytical, tol::kNano) << "w = " << w.transpose();
  }
}

TEST(SO3Test, SO3LogMulExpDerivativeNearZero) {
  // test small angle cases
  // clang-format off
  const std::vector<Vector<double, 3>> samples = {
    {0.0, 0.0, 0.0},
    {-1.0e-5, 1.0e-5, 0.3e-5},
    {0.1e-5, 0.0, -0.1e-5},
    {-0.2e-8, 0.3e-7, 0.0},
  };
  // clang-format on

  // for small hangle to hold, R should be identity
  const Quaternion<double> R = Quaternion<double>::Identity();

  // functor that holds R fixed and multiplies on w
  const auto fix_r_functor = [&](const Vector<double, 3>& w) -> Vector<double, 3> {
    return math::RotationLog(R * math::QuaternionExp(w));
  };

  for (const auto& w : samples) {
    const Matrix<double, 3, 3> J_analytical = math::SO3LogMulExpDerivative(R, w);
    const Matrix<double, 3, 3> J_numerical = NumericalJacobian(w, fix_r_functor);
    ASSERT_EIGEN_NEAR(J_numerical, J_analytical, tol::kNano);
  }
}

}  // namespace math
