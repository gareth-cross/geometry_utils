// Copyright 2020 Gareth Cross
#pragma once
#include <functional>
#include <type_traits>

#include "geometry_utils/manifold.hpp"
#include "geometry_utils/matrix_types.hpp"

// Turn off warning about constant if statements.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif  // _MSC_VER

namespace math {

// Numerically compute first derivative of f(x) via central difference.
// Uses the third-order approximation, which has error in O(h^6).
//
// References:
// http://www.rsmas.miami.edu/personal/miskandarani/Courses/MSC321/lectfiniteDifference.pdf
// https://en.wikipedia.org/wiki/Finite_difference_coefficient
template <typename Tx, typename Th, typename Function>
auto NumericalDerivative2(const Tx x, const Th h, Function func) -> decltype(func(x, h)) {
  using ResultType = decltype(func(x, h));
  const Th h2 = h * 2;
  const Th h3 = h * 3;
  const ResultType c1 = func(x, h) - func(x, -h);
  const ResultType c2 = func(x, h2) - func(x, -h2);
  const ResultType c3 = func(x, h3) - func(x, -h3);
  return (c1 * 45 - c2 * 9 + c3) / (60 * h);
}

// Version of numericalDerivative2 that accepts a unary function.
template <typename Tx, typename Th, typename Function>
auto NumericalDerivative(const Tx x, const Th h, Function func) -> decltype(func(x)) {
  return NumericalDerivative2(x, h, [&func](const Tx x, const Th dx) { return func(x + dx); });
}

// Numerically compute the jacobian of vector function `y = f(x)` via the
// central-difference.
template <typename XExpr, typename Function>
auto NumericalJacobian(const XExpr& x, Function func, const double h = 0.01)
    -> Matrix<typename Manifold<XExpr>::Scalar, Manifold<decltype(func(x))>::Dim,
              Manifold<XExpr>::Dim> {
  using YExpr = typename std::decay<decltype(func(x))>::type;
  constexpr int DimX = Manifold<XExpr>::Dim;
  constexpr int DimY = Manifold<YExpr>::Dim;
  using Scalar = typename Manifold<XExpr>::Scalar;

  // Compute the output expression at the linearization point.
  const YExpr y_0 = func(x);

  // Possibly allocate for the result, since dimensions may be dynamic.
  Matrix<Scalar, DimY, DimX> J;
  if (DimX == Eigen::Dynamic || DimY == Eigen::Dynamic) {
    J.resize(Manifold<YExpr>::TangentDimension(y_0), Manifold<XExpr>::TangentDimension(x));
  }

  // Pre-allocate `delta` once and re-use it.
  Vector<Scalar, DimX> delta;
  if (DimX == Eigen::Dynamic) {
    delta.resize(Manifold<XExpr>::TangentDimension(x));
  }

  for (int j = 0; j < Manifold<XExpr>::TangentDimension(x); ++j) {
    // Take derivative wrt the j'th dimension of X
    const auto wrapped = [&](const XExpr& x, Scalar dx) {
      // apply perturbation in the tangent space
      delta.setZero();
      delta[j] = dx;
      // Perform the operation: x [+] f(dx), where [+] is the manifold composition.
      const auto x_plus_dx = Manifold<XExpr>::To(x, delta);
      const auto y = func(x_plus_dx);
      // determine the perturbation in y: dy = f^-1(y^-1 [+] y)
      // where f() maps to and from the manifold
      return Manifold<YExpr>::From(y_0, y);
    };
    J.col(j) = NumericalDerivative2(x, static_cast<Scalar>(h), wrapped);
  }
  return J;
}

}  // namespace math

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER
