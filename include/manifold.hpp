// Copyright 2020 Gareth Cross
#pragma once
#include <type_traits>

#include "matrix_types.hpp"
#include "so3.hpp"

namespace math {

// True if the argument is a 'scalar' type.
template <typename T>
struct IsScalar : public std::is_floating_point<T> {};

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

  // Get runtime dimension of the tangent space of an object.
  static constexpr int TangentDimension(const T& x) {
    (void)x;  //  silence un-referenced parameter in MSVC
    return Dim;
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

  // Runtime dimension of vector.
  static int TangentDimension(const T& x) { return static_cast<int>(x.rows()); }
};

// Traits on floats/doubles.
template <typename T>
struct Manifold<T, typename std::enable_if<IsScalar<T>::value>::type> {
  static constexpr int Dim = 1;
  using Scalar = T;
  using VectorType = Vector<Scalar, 1>;

  // Defined so numericalDerivative works.
  static VectorType From(const T& x, const T& y) { return VectorType{y - x}; }

  // Defined so numericalDerivative works.
  template <typename Derived>
  static T To(const T& x, const Eigen::MatrixBase<Derived>& v) {
    // If we can check at compile time, enforce this is a 1x1 matrix.
    static_assert(Eigen::MatrixBase<Derived>::RowsAtCompileTime == Eigen::Dynamic ||
                      Eigen::MatrixBase<Derived>::RowsAtCompileTime == 1,
                  "Must be a scalar");
    static_assert(Eigen::MatrixBase<Derived>::ColsAtCompileTime == Eigen::Dynamic ||
                      Eigen::MatrixBase<Derived>::ColsAtCompileTime == 1,
                  "Must be a scalar");
    return x + static_cast<T>(v[0]);
  }

  // Runtime dimension of vector.
  static constexpr int TangentDimension(const T& x) {
    (void)x;
    return Dim;
  }
};

}  // namespace math
