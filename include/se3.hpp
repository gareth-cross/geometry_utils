#pragma once
#include "matrix_types.hpp"
#include "so3.hpp"

namespace math {

// Simple pose type with rotation and translation.
template <typename Scalar>
struct SE3 {
  // Rotation. Quaternion corresponding to upper-left 3x3 block of SE(3) matrix.
  Quaternion<Scalar> R;

  // Translation. Vector corresponding to upper-right 3x3 block of SE(3) matrix.
  Vector<Scalar, 3> t;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//
template <typename Scalar>
SE3<Scalar> SE3Exp(const Vector<Scalar, 6>& w_u) {
  SE3<Scalar> result;
  // TODO(gareth): Share logic here.
  result.R = math::QuaternionExp(w_u.template head<3>());
  result.t = math::SO3Jacobian(w_u.template head<3>()) * w_u.template tail<3>();
  return result;
}

}  // namespace math
