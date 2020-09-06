#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace math {

template <typename T, int Rows = Eigen::Dynamic>
using Vector = Eigen::Matrix<T, Rows, 1>;

template <typename T, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
using Matrix = Eigen::Matrix<T, Rows, Cols>;

template <typename T>
using Quaternion = Eigen::Quaternion<T>;

// Templates for extracting useful values from eigen types.
template <typename Derived>
using ScalarType = typename Eigen::MatrixBase<Derived>::Scalar;

template <typename Derived>
using BaseType = typename Eigen::MatrixBase<Derived>;

// Check if something is an Eigen quaternion.
template <typename Type>
struct IsQuaternion : public std::false_type {};

template <typename Scalar, int Options>
struct IsQuaternion<Eigen::Quaternion<Scalar, Options>> : public std::true_type {};

// Check if something is an Eigen vector.
template <typename Type, typename = void>
struct IsVector {
  static constexpr bool value = false;
};
template <typename Type>
struct IsVector<Type, decltype(Type::IsVectorAtCompileTime, void())> {
  static constexpr bool value = Type::IsVectorAtCompileTime;
};

//
// Compile time tests.
//

static_assert(IsQuaternion<Eigen::Quaternionf>::value, "");
static_assert(IsQuaternion<Eigen::Quaterniond>::value, "");
static_assert(!IsQuaternion<Eigen::Vector3d>::value, "");

static_assert(IsVector<Eigen::Vector3f>::value, "");
static_assert(IsVector<Eigen::VectorXd>::value, "");
static_assert(!IsVector<Eigen::Matrix3f>::value, "");

}  // namespace math
