// Copyright 2020 Gareth Cross
#include "geometry_utils/rotation_utils.hpp"

#include <chrono>
#include <random>

#include "geometry_utils/numerical_derivative.hpp"
#include "test_utils.hpp"

namespace math {

TEST(SO3Test, TestBasisFromZAxisWithMinAngularXY) {
  std::default_random_engine engine{0};  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution<double> distribution{-1., 1.};
  for (const auto& random_vector : kRandomRotationVectorsZero2Pi) {
    // Create an old basis whose z-vector we want to align to:
    const math::Matrix<double, 3, 3> R_basis_old = math::QuaternionExp(random_vector).matrix();

    // The z-vector we want to align our basis' z vector to.
    const Eigen::Vector3d basis_new_z =
        Eigen::Vector3d{distribution(engine), distribution(engine), distribution(engine)}
            .normalized();

    // Solve for the new basis
    const auto target_x = R_basis_old.col(0);
    const auto target_y = R_basis_old.col(1);
    const math::Matrix<double, 3, 3> R_output =
        math::BasisFromZAxisWithMinAngularXY<double>(basis_new_z, target_x, target_y);

    // Check that the z-vector does indeed match
    ASSERT_EIGEN_NEAR(basis_new_z, R_output.col(2), tol::kPico);

    // Check that the resulting matrix is orthonormal
    ASSERT_EIGEN_NEAR(Eigen::Matrix3d::Identity(), R_output.transpose() * R_output, tol::kPico);
    ASSERT_NEAR(1.0, R_output.determinant(), tol::kPico);

    // Check that we actually maximized the quantity we wanted:
    const auto cost = [=](double theta) -> double {
      // Rotate a bit about the z-axis (in the plane)
      const math::Matrix<double, 3, 3> R_perturbed =
          R_output * math::QuaternionExp(Eigen::Vector3d::UnitZ() * theta).matrix();
      return R_perturbed.col(0).dot(target_x) + R_perturbed.col(1).dot(target_y);
    };
    const auto cost_derivative = [&](double theta) -> double {
      return math::NumericalDerivative(theta, 0.01, cost);
    };

    const double cost_D_theta = math::NumericalDerivative(0.0, 0.01, cost);
    const double cost_D2_theta = math::NumericalDerivative(0.0, 0.01, cost_derivative);
    ASSERT_NEAR(0.0, cost_D_theta, tol::kNano);
    ASSERT_LT(cost_D2_theta, 0.0);

    // Also test that when given a basis, it returns that basis
    ASSERT_EIGEN_NEAR(R_basis_old,
                      math::BasisFromZAxisWithMinAngularXY<double>(
                          R_basis_old.col(2), R_basis_old.col(0), R_basis_old.col(1)),
                      tol::kNano);
  }

  // Try some matrices that are reflections:
  for (const auto& random_vector : kRandomRotationVectorsZero2Pi) {
    // Create an old basis whose z-vector we want to align to:
    const math::Matrix<double, 3, 3> R_basis_old = math::QuaternionExp(random_vector).matrix();

    // Just scale z by negative one to reflect it:
    const Eigen::Vector3d basis_new_z = R_basis_old.col(2) * -1;

    // Solve for the new basis
    const auto target_x = R_basis_old.col(0);
    const auto target_y = R_basis_old.col(1);
    const math::Matrix<double, 3, 3> R_output =
        math::BasisFromZAxisWithMinAngularXY<double>(basis_new_z, target_x, target_y);

    // Check that the z-vector does indeed match
    ASSERT_EIGEN_NEAR(basis_new_z, R_output.col(2), tol::kPico);

    // Check that the resulting matrix is orthonormal
    ASSERT_EIGEN_NEAR(Eigen::Matrix3d::Identity(), R_output.transpose() * R_output, tol::kPico);
    ASSERT_NEAR(1.0, R_output.determinant(), tol::kPico);

    // Check that we actually maximized the quantity we wanted:
    const auto cost = [=](double theta) -> double {
      // Rotate a bit about the z-axis (in the plane)
      const math::Matrix<double, 3, 3> R_perturbed =
          R_output * math::QuaternionExp(Eigen::Vector3d::UnitZ() * theta).matrix();
      return R_perturbed.col(0).dot(target_x) + R_perturbed.col(1).dot(target_y);
    };
    const auto cost_derivative = [&](double theta) -> double {
      return math::NumericalDerivative(theta, 0.01, cost);
    };

    // We don't check for maxima here, since there are two.
    const double cost_D_theta = math::NumericalDerivative(0.0, 0.01, cost);
    ASSERT_NEAR(0.0, cost_D_theta, tol::kNano);
  }

  // Try some rotations by 90 degrees:
  for (const auto& random_vector : kRandomRotationVectorsZero2Pi) {
    // Create an old basis whose z-vector we want to align to:
    const math::Matrix<double, 3, 3> R_basis_old = math::QuaternionExp(random_vector).matrix();

    // Just scale z by negative one to reflect it:
    const Eigen::Vector3d basis_new_z =
        math::QuaternionExp(math::Vector<double, 3>::UnitX() * M_PI / 2).matrix() *
        R_basis_old.col(2);

    // Solve for the new basis
    const auto target_x = R_basis_old.col(0);
    const auto target_y = R_basis_old.col(1);
    const math::Matrix<double, 3, 3> R_output =
        math::BasisFromZAxisWithMinAngularXY<double>(basis_new_z, target_x, target_y);

    // Check that the z-vector does indeed match
    ASSERT_EIGEN_NEAR(basis_new_z, R_output.col(2), tol::kPico);

    // Check that the resulting matrix is orthonormal
    ASSERT_EIGEN_NEAR(Eigen::Matrix3d::Identity(), R_output.transpose() * R_output, tol::kPico);
    ASSERT_NEAR(1.0, R_output.determinant(), tol::kPico);

    // Check that we actually maximized the quantity we wanted:
    const auto cost = [=](double theta) -> double {
      // Rotate a bit about the z-axis (in the plane)
      const math::Matrix<double, 3, 3> R_perturbed =
          R_output * math::QuaternionExp(Eigen::Vector3d::UnitZ() * theta).matrix();
      return R_perturbed.col(0).dot(target_x) + R_perturbed.col(1).dot(target_y);
    };
    const auto cost_derivative = [&](double theta) -> double {
      return math::NumericalDerivative(theta, 0.01, cost);
    };

    const double cost_D_theta = math::NumericalDerivative(0.0, 0.01, cost);
    const double cost_D2_theta = math::NumericalDerivative(0.0, 0.01, cost_derivative);
    ASSERT_NEAR(0.0, cost_D_theta, tol::kNano);
    ASSERT_LT(cost_D2_theta, 0.0);
  }
}

}  // namespace math
