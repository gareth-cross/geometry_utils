// Copyright 2020 Gareth Cross
#pragma once
#include <type_traits>
#include "matrix_types.hpp"
#include "so3.hpp"

namespace math {

// Unspecified manifold traits.
template <typename T, typename = void>
struct Manifold {};

// Operations on quaternions.
template <typename T>
struct Manifold<T, typename std::enable_if<IsQuaternion<T>::value>::type> {
  // Quaternions are convertible to so(3).
  static constexpr int Dim = 3;
  using Scalar = typename T::Scalar;
  using VectorType = Vector<Scalar, Dim>;

  // Map from the manifold to a vector in the tangent space of `x`.
  static VectorType From(const T& x, const T& y) { return RotationLog(x.conjugate() * y); }

  // Map a vector to the manifold in the tangent space of `x`.
  template <typename Derived>
  static T To(const T& x, const Eigen::MatrixBase<Derived>& v) {
    return x * QuaternionExp(v);
  }
};

// Traits on vector types.
template <typename T>
struct Manifold<T, typename std::enable_if<IsVector<T>::value>::type> {
  // Inherit dimensionality from the vector itself.
  static constexpr int Dim = Eigen::MatrixBase<T>::RowsAtCompileTime;
  using Scalar = typename T::Scalar;
  using VectorType = Vector<Scalar, Dim>;

  // Defined so numericalDerivative works.
  static VectorType From(const VectorType& x, const VectorType& y) { return y - x; }

  // Defined so numericalDerivative works.
  template <typename Derived>
  static VectorType To(const T& x, const Eigen::MatrixBase<Derived>& v) {
    return x + v;
  }
};

}  // namespace math
