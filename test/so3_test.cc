// Copyright 2020 Gareth Cross
#include "geometry_utils/so3.hpp"

#include <chrono>
#include <random>

#include "geometry_utils/numerical_derivative.hpp"
#include "test_utils.hpp"

// Disable cast warning for the purpose of this test on MSVC.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4305)
#endif  // _MSC_VER

// TODO(gareth): I think gtest has some tools for testing both double/float.
// Should probably use those here instead of calling manually.
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
    const Quaternion<Scalar> q = math::QuaternionExp(w);
    const Matrix<Scalar, 4, 3> q_D_w = QuaternionExpJacobian(w);

    // compare to the exponential map as a power series ~ 50 terms
    ASSERT_EIGEN_NEAR(ExpMatrixSeries(Skew3(w), 50), q.matrix(), matrix_tol)
        << "w = " << w.transpose();

    // compare to Eigen implementation for good measure
    const Eigen::AngleAxis<Scalar> aa(w.norm(), w.normalized());
    ASSERT_EIGEN_NEAR(aa.toRotationMatrix(), q.matrix(), matrix_tol) << "w = " << w.transpose();

    // check derivative numerically
    const Matrix<Scalar, 4, 3> J_numerical =
        NumericalJacobian(w, [](const Vector<Scalar, 3>& w) -> Vector<Scalar, 4> {
          // convert to correct order here
          const Quaternion<Scalar> q = math::QuaternionExp(w);
          return Vector<Scalar, 4>(q.w(), q.x(), q.y(), q.z());
        });
    ASSERT_EIGEN_NEAR(J_numerical, q_D_w, deriv_tol) << "w = " << w.transpose();
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
    for (const Eigen::Vector3d& w : kRandomRotationVectorsZero2Pi) {
      TestOmega<double>(w, tol::kPico, tol::kNano);
      TestOmega<float>(w.cast<float>(), tol::kMicro, tol::kMilli / 10);
    }
  }

  void TestNearZero() const {
    TestOmega<double>({1.0e-7, 0.5e-7, 3.5e-8}, tol::kNano, tol::kMicro);
    TestOmega<double>({0.0, 0.0, 0.0}, tol::kNano, tol::kMicro);
    TestOmega<float>({1.0e-7, 0.5e-7, 3.5e-8}, tol::kNano, tol::kMicro);
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
  for (const Eigen::Vector3d& w : kRandomRotationVectorsZero2Pi) {
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
  static void TestJacobian(const Vector<Scalar, 3>& w_a, const Scalar deriv_tol,
                           const Scalar invert_tol) {
    const Matrix<Scalar, 3, 3> J_analytical = math::SO3Jacobian(w_a);
    // This jacobian is only valid for small `w`, so evaluate about zero.
    const Matrix<Scalar, 3, 3> J_numerical =
        NumericalJacobian(Vector<Scalar, 3>::Zero(),
                          [&](const Vector<Scalar, 3>& w) { return QuaternionExp(w_a + w); });
    EXPECT_EIGEN_NEAR(J_numerical, J_analytical, deriv_tol) << "w = " << w_a.transpose();

    // Test that this is the inverse of the its corresponding method.
    if (invert_tol >= 0) {
      EXPECT_EIGEN_NEAR(J_analytical, math::SO3JacobianInverse(w_a).inverse(), invert_tol);
    }
  }

  template <typename Scalar>
  static void TestMatchesSE3VMatrix(const Vector<Scalar, 3>& w,
                                    const Vector<Scalar, 3>& u,  // translational delta
                                    const Scalar tol) {
    // evaluate the SE(3) exponential map numerically
    const Matrix<Scalar, 4, 4> A = (Matrix<Scalar, 4, 4>() << Skew3(w), u, 0, 0, 0, 0).finished();
    const Matrix<Scalar, 4, 4> exp_se3 = ExpMatrixSeries(A, 50);

    // Check that this matches the matrix `V` from the SE(3) exponential map.
    const Matrix<Scalar, 3, 1> t_on_group = exp_se3.template block<3, 1>(0, 3);
    EXPECT_EIGEN_NEAR(t_on_group, math::SO3Jacobian(-w) * u, tol);
  }

  void TestGeneral() {
    for (const Eigen::Vector3d& w : kRandomRotationVectorsZero2Pi) {
      TestJacobian<double>(w, tol::kNano, tol::kNano);
      TestJacobian<float>(w.cast<float>(), tol::kMilli / 10, 0.0005);

      // check that this correctly executes the left jacobian of SO(3) as required to
      // compute the exponential map of SE(3)
      TestMatchesSE3VMatrix<double>(w, {-1.5, 0.4, 3.0}, tol::kNano);
    }
  }

  void TestNearZero() {
    // test small angle cases
    // clang-format off
    const std::vector<Vector<double, 3>> samples = {
      {0.0, 0.0, 0.0},
      {-1.0e-7, 1.0e-7, 0.3e-7},
      {0.1e-10, 0.0, -0.1e-9},
      {-0.2e-8, 0.3e-7, 0.0},
      {-0.2312e-9, 0.0, 0.1153e-7},
      {1.0e-12, 2.0e-12, 0.0},
    };
    // clang-format on
    for (const auto& w : samples) {
      TestJacobian<double>(w, tol::kNano, -1);
      TestJacobian<float>(w.cast<float>(), tol::kMicro, -1);
    }
  }
};

TEST_FIXTURE(TestSO3Jacobian, TestGeneral)
TEST_FIXTURE(TestSO3Jacobian, TestNearZero)

class TestSO3DerivativeInverse : public ::testing::Test {
 public:
  template <typename Scalar>
  static Vector<Scalar, 9> VecExpMatrix(const Vector<Scalar, 3>& w) {
    // Convert to vectorized format.
    const Matrix<Scalar, 3, 3> R = math::QuaternionExp(w).matrix();
    return Eigen::Map<const Vector<Scalar, 9>>(R.data());
  }

  template <typename Scalar>
  static void TestDerivative(const Vector<Scalar, 3>& w, const Scalar deriv_tol) {
    const Matrix<Scalar, 3, 3> J_analytical = math::SO3JacobianInverse(w);
    // This jacobian is only valid for small `dw`, so evaluate about zero.
    const Matrix<Scalar, 3, 3> J_numerical =
        NumericalJacobian(Vector<Scalar, 3>::Zero(), [&](const Vector<Scalar, 3>& dw) {
          return RotationLog(QuaternionExp(w) * QuaternionExp(dw));
        });
    ASSERT_EIGEN_NEAR(J_numerical, J_analytical, deriv_tol) << "w = " << w.transpose();

    // the inverse of this should be the derivative of: log[exp(-w) * exp(w + dw)]
    // because:
    //    exp(v + dv) = exp(w) * exp(dw)
    // so:
    //    v + dv = log[exp(w) * exp(dw)]
    // if we say:
    //    dv ~= J * dw (to first order)
    // then:
    //    dw ~= J^-1 dv
    const Matrix<Scalar, 3, 3> J_numerical_2 =
        NumericalJacobian(Vector<Scalar, 3>::Zero(),
                          [&](const Vector<Scalar, 3>& dw) { return QuaternionExp(w + dw); });
    ASSERT_EIGEN_NEAR(J_numerical_2.inverse(), J_analytical, deriv_tol);
  }

  void TestGeneral() {
    for (const Eigen::Vector3d& w : kRandomRotationVectorsZeroPi) {
      TestDerivative<double>(w, tol::kNano / 10);
      TestDerivative<float>(w.cast<float>(), tol::kMilli / 10);
    }
  }

  void TestNearZero() {
    // test small angle cases
    // clang-format off
    const std::vector<Vector<double, 3>> samples = {
      {0.0, 0.0, 0.0},
      {-1.0e-7, 1.0e-7, 0.3e-7},
      {0.1e-10, 0.0, -0.1e-9},
      {-0.2e-8, 0.3e-7, 0.0},
      {-0.2312e-9, 0.0, 0.1153e-7},
      {1.0e-12, 2.0e-12, 0.0},
    };
    // clang-format on
    for (const auto& w : samples) {
      TestDerivative<double>(w, tol::kNano);
      TestDerivative<float>(w.cast<float>(), tol::kMicro);
    }
  }
};

TEST_FIXTURE(TestSO3DerivativeInverse, TestGeneral)
TEST_FIXTURE(TestSO3DerivativeInverse, TestNearZero)

class TestSO3FromEulerAngles : public ::testing::Test {
 public:
  template <typename Scalar>
  static void TestDerivative(const Vector<Scalar, 3>& xyz, const Scalar deriv_tol) {
    for (const CompositionOrder order : {CompositionOrder::ZYX, CompositionOrder::XYZ}) {
      const auto q_and_deriv = math::SO3FromEulerAngles(xyz, order);

      // check that this matches Eigen
      if (order == CompositionOrder::ZYX) {
        const Matrix<Scalar, 3, 3> R_eigen =
            Eigen::AngleAxis<Scalar>(xyz[2], Vector<Scalar, 3>::UnitZ()).matrix() *
            Eigen::AngleAxis<Scalar>(xyz[1], Vector<Scalar, 3>::UnitY()).matrix() *
            Eigen::AngleAxis<Scalar>(xyz[0], Vector<Scalar, 3>::UnitX()).matrix();
        ASSERT_EIGEN_NEAR(R_eigen, q_and_deriv.q.matrix(), deriv_tol);
      } else {
        const Matrix<Scalar, 3, 3> R_eigen =
            Eigen::AngleAxis<Scalar>(xyz[0], Vector<Scalar, 3>::UnitX()).matrix() *
            Eigen::AngleAxis<Scalar>(xyz[1], Vector<Scalar, 3>::UnitY()).matrix() *
            Eigen::AngleAxis<Scalar>(xyz[2], Vector<Scalar, 3>::UnitZ()).matrix();
        ASSERT_EIGEN_NEAR(R_eigen, q_and_deriv.q.matrix(), deriv_tol);
      }

      // Check derivative of tangent-space wrt the euler angles.
      const Matrix<Scalar, 3, 3> J_numerical = NumericalJacobian(
          xyz,
          [&](const Vector<Scalar, 3>& xyz) { return math::SO3FromEulerAngles(xyz, order).q; });
      ASSERT_EIGEN_NEAR(J_numerical, q_and_deriv.rotation_D_angles, deriv_tol)
          << "xyz = " << xyz.transpose();
    }
  }

  void TestGeneral() {
    for (const Eigen::Vector3d& w : kRandomRotationVectorsZero2Pi) {
      TestDerivative<double>(w, tol::kNano);
      TestDerivative<float>(w.cast<float>(), 1.0e-4);
    }
  }
};

TEST_FIXTURE(TestSO3FromEulerAngles, TestGeneral)

TEST(SO3Test, TestEulerAnglesFromSO3) {
  // Test some conversions to and from euler angles.
  const std::vector<Eigen::Vector3d> angles = {
      {-0.5, 1.2, 0.3}, {0.0, 0.0, 0.0},  {-0.3, 0.6, -0.9},   {1.4, 0.0, -1.3},
      {0.0, 0.0, 1.2},  {0.0, -0.9, 0.0}, {0.0, 0.0, 0.74123},
  };
  // test that we can go both directions
  for (const auto& xyz : angles) {
    const math::SO3FromEulerAngles_<double> rot =
        math::SO3FromEulerAngles(xyz, CompositionOrder::ZYX);
    ASSERT_EIGEN_NEAR(xyz, math::EulerAnglesFromSO3(rot.q), tol::kNano);
  }
}

// Test derivative for rotating a point.
TEST(SO3Test, RotateVectorSO3TangentJacobian) {
  const std::vector<Eigen::Vector3d> points = {
      {0, 0, 0},
      {1.0, -5.0, 3.0},
      {0.02, -1.2, 0.02},
  };
  for (const auto& w : kRandomRotationVectorsZero2Pi) {
    for (const auto& p : points) {
      // compute jacobian analytically
      const Matrix<double, 3, 3> J_analytical =
          math::RotateVectorSO3TangentJacobian(QuaternionExp(w), p);
      // compute numerically
      const Matrix<double, 3, 3> J_numerical = NumericalJacobian(
          Eigen::Vector3d::Zero(), [&](const Vector<double, 3>& dw) -> Vector<double, 3> {
            return (QuaternionExp(w) * QuaternionExp(dw)).matrix() * p;
          });
      ASSERT_EIGEN_NEAR(J_analytical, J_numerical, tol::kPico);
    }
  }
}

// Test the derivative of the exponential map, matrix form.
class TestMatrixExpJacobian : public ::testing::Test {
 public:
  template <typename Scalar>
  static Vector<Scalar, 9> VecExpMatrix(const Vector<Scalar, 3>& w) {
    // Convert to vectorized format.
    const Matrix<Scalar, 3, 3> R = math::QuaternionExp(w).matrix();
    return Eigen::Map<const Vector<Scalar, 9>>(R.data());
  }

  template <typename Scalar>
  static void TestDerivative(const Vector<Scalar, 3>& w, const Scalar deriv_tol) {
    const Matrix<Scalar, 9, 3> D_w = math::SO3ExpMatrixJacobian(w);
    const Matrix<Scalar, 9, 3> J_numerical =
        NumericalJacobian(w, &TestMatrixExpJacobian::VecExpMatrix<Scalar>);
    ASSERT_EIGEN_NEAR(J_numerical, D_w, deriv_tol);
  }

  void TestGeneral() {
    for (const Eigen::Vector3d& w : kRandomRotationVectorsZero2Pi) {
      TestDerivative<double>(w, tol::kNano / 10);
      TestDerivative<float>(w.cast<float>(), tol::kMilli / 10);
    }
  }

  void TestNearZero() {
    TestDerivative<double>({-1.0e-7, 1.0e-8, 0.5e-7}, tol::kMicro);
    TestDerivative<float>({-1.0e-7, 1.0e-8, 0.5e-7}, tol::kMicro);

    // at exactly zero it should be identically equal to the generators of SO(3)
    const Matrix<double, 9, 3> J_at_zero =
        math::SO3ExpMatrixJacobian(Vector<double, 3>::Zero().eval());
    const auto i_hat = Vector<double, 3>::UnitX();
    const auto j_hat = Vector<double, 3>::UnitY();
    const auto k_hat = Vector<double, 3>::UnitZ();
    EXPECT_EIGEN_NEAR(Skew3(-i_hat), J_at_zero.block(0, 0, 3, 3), tol::kPico);
    EXPECT_EIGEN_NEAR(Skew3(-j_hat), J_at_zero.block(3, 0, 3, 3), tol::kPico);
    EXPECT_EIGEN_NEAR(Skew3(-k_hat), J_at_zero.block(6, 0, 3, 3), tol::kPico);
  }
};

TEST_FIXTURE(TestMatrixExpJacobian, TestGeneral)
TEST_FIXTURE(TestMatrixExpJacobian, TestNearZero)

// Have to be careful when testing this method numerically, since the output of log() can
// jump around if the rotation R * exp(w) is large.
TEST(SO3Test, SO3LogMulExpJacobian) {
  // create the matrix R we multiply against
  const Vector<double, 3> R_log{0.21, -0.25, 0.1};
  const Quaternion<double> R = math::QuaternionExp(R_log);

  // functor that holds R fixed and multiplies on w
  const auto fix_r_functor = [&](const Vector<double, 3>& w) -> Vector<double, 3> {
    return math::RotationLog(R * math::QuaternionExp(w));
  };

  // do a bunch of random ones too
  for (const auto& w : kRandomRotationVectorsZeroPi) {
    const Matrix<double, 3, 3> J_analytical = math::SO3LogMulExpJacobian(R, w);
    const Matrix<double, 3, 3> J_numerical = NumericalJacobian(w, fix_r_functor);
    ASSERT_EIGEN_NEAR(J_numerical, J_analytical, tol::kNano) << "w = " << w.transpose();
  }

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
    const Matrix<double, 3, 3> J_analytical = math::SO3LogMulExpJacobian(R, w);
    const Matrix<double, 3, 3> J_numerical = NumericalJacobian(w, fix_r_functor);
    ASSERT_EIGEN_NEAR(J_numerical, J_analytical, tol::kNano) << "w = " << w.transpose();
  }
}

TEST(SO3Test, SO3LogMulExpJacobianNearZero) {
  // test small angle cases
  // clang-format off
  const std::vector<Vector<double, 3>> samples = {
    {0.0, 0.0, 0.0},
    {-1.0e-7, 1.0e-7, 0.3e-7},
    {0.1e-10, 0.0, -0.1e-9},
    {-0.2e-8, 0.3e-7, 0.0},
  };
  // clang-format on

  // for small angle to hold, R should be identity
  const Quaternion<double> R = Quaternion<double>::Identity();

  // functor that holds R fixed and multiplies on w
  const auto fix_r_functor = [&](const Vector<double, 3>& w) -> Vector<double, 3> {
    return math::RotationLog(R * math::QuaternionExp(w));
  };

  for (const auto& w : samples) {
    const Matrix<double, 3, 3> J_analytical = math::SO3LogMulExpJacobian(R, w);
    const Matrix<double, 3, 3> J_numerical = NumericalJacobian(w, fix_r_functor);
    ASSERT_EIGEN_NEAR(J_numerical, J_analytical, tol::kNano);
  }
}

}  // namespace math

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER
