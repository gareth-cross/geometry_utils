// Copyright 2020 Gareth Cross
#pragma once
#include <gtest/gtest.h>
#include <Eigen/Dense>

// Numerical tolerances for tests.
namespace tol {
static constexpr double kDeci = 1.0e-1;
static constexpr double kCenti = 1.0e-2;
static constexpr double kMilli = 1.0e-3;
static constexpr double kMicro = 1.0e-6;
static constexpr double kNano = 1.0e-9;
static constexpr double kPico = 1.0e-12;
}  // namespace tol

// Print variable w/ name.
#define PRINT(x) printImpl(#x, x)

template <typename Xpr>
void printImpl(const std::string& name, Xpr xpr) {
  std::cout << name << "=" << xpr << std::endl;
}

// Define a test on a class.
#define TEST_FIXTURE(object, function) \
  TEST_F(object, function) { function(); }

// Macro to compare eigen matrices and print a nice error.
#define EXPECT_EIGEN_NEAR(a, b, tol) EXPECT_PRED_FORMAT3(math::expectEigenNear, a, b, tol)
#define ASSERT_EIGEN_NEAR(a, b, tol) ASSERT_PRED_FORMAT3(math::expectEigenNear, a, b, tol)

namespace math {

// 300 randomly generated rotation vectors (range 0 to 2pi), used for testing.
extern const std::vector<Eigen::Vector3d> kRandomRotationVectorsZero2Pi;

// 300 more in 0 to pi.
extern const std::vector<Eigen::Vector3d> kRandomRotationVectorsZeroPi;

// Compare two eigen matrices. Use EXPECT_EIGEN_NEAR()
template <typename Ta, typename Tb>
testing::AssertionResult expectEigenNear(const std::string& name_a, const std::string& name_b,
                                         const std::string& name_tol,
                                         const Eigen::MatrixBase<Ta>& a,
                                         const Eigen::MatrixBase<Tb>& b, double tolerance) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return testing::AssertionFailure()
           << "Dimensions of " << name_a << " and " << name_b << " do not match.";
  }
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) {
      const double delta = a(i, j) - b(i, j);
      if (std::abs(delta) > tolerance || std::isnan(delta)) {
        const std::string index_str = "(" + std::to_string(i) + ", " + std::to_string(j) + ")";
        return testing::AssertionFailure()
               << "Matrix equality " << name_a << " == " << name_b << " failed because:\n"
               << name_a << index_str << " - " << name_b << index_str << " = " << delta << " > "
               << name_tol << "\nWhere " << name_a << " evaluates to:\n"
               << a << "\n and " << name_b << " evaluates to:\n"
               << b << "\n and " << name_tol << " evaluates to: " << tolerance << "\n";
      }
    }
  }
  return testing::AssertionSuccess();
}

/// Create a vector in the specified range.
/// Begins at `start` and increments by `step` until >= `end`.
std::vector<double> Range(double start, double end, double step);
/**
 * @brief Exponential map via power series. Computes the value of exp(A), where A is a square
 *  matrix.
 * @note Refer to:
 *  "Chapter 4: Basics of Classical Lie Groups: The Exponential Map,
 *   Lie Groups, and Lie Algebras" - Jean Gallier
 * @param num_terms Number of terms in power series. Note that large values
 *  will result in floating-point underflow, so be careful.
 */
template <typename Scalar, int Rows, int Cols>
Eigen::Matrix<Scalar, Rows, Cols> ExpMatrixSeries(const Eigen::Matrix<Scalar, Rows, Cols>& A,
                                                  const int num_terms = 15) {
  Eigen::Matrix<double, Rows, Cols> A_power, solution;
  A_power.setIdentity();
  double fac_accum = 1;
  solution.setIdentity();  //  term 0
  for (int p = 1; p < num_terms; ++p) {
    A_power *= A.template cast<double>();
    fac_accum *= p;
    solution.noalias() += A_power / fac_accum;
  }
  return solution.template cast<Scalar>();
}

}  // namespace math
