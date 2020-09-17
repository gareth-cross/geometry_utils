// Copyright 2020 Gareth Cross
#pragma once
#include "internal_utils.hpp"
#include "matrix_types.hpp"

/*
 * Some useful functions for SO(3) and its derivatives. I am only bothering to define
 * the right-side jacobians, since that is why I use.
 *
 * Tests in so3_test.cc
 *
 * TODO(gareth): Potentially add the jacobians for rotation composition on SO(3), although
 * they are fairly simple.
 */
namespace math {

/**
 * Convert 3-vector `v` into 3x3 skew symmetric matrix. This operator is
 * often denoted as [v]_x, since it acts as the cross-product when left-multipled.
 *
 * See: https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product
 */
template <typename Derived>
Matrix<ScalarType<Derived>, 3, 3> Skew3(const Eigen::MatrixBase<Derived>& v_xpr) {
  using Scalar = ScalarType<Derived>;
  const Vector<Scalar, 3>& v = v_xpr.eval();
  Matrix<ScalarType<Derived>, 3, 3> S;
  S.diagonal().setZero();
  S(0, 1) = -v[2];
  S(0, 2) = v[1];
  S(1, 0) = v[2];
  S(1, 2) = -v[0];
  S(2, 0) = -v[1];
  S(2, 1) = v[0];
  return S;
}

/**
 * Return the 4x4 matrix that executes a quaternion multiplication. Assumes elements
 * are ordered [w, x, y, z].
 */
template <typename Derived>
Matrix<typename Eigen::QuaternionBase<Derived>::Scalar, 4, 4> QuaternionMulMatrix(
    const Eigen::QuaternionBase<Derived>& q) {
  Matrix<typename Eigen::QuaternionBase<Derived>::Scalar, 4, 4> Q;
  Q.diagonal().array() = q.w();
  // top right
  Q(0, 1) = -q.x();
  Q(0, 2) = -q.y();
  Q(0, 3) = -q.z();
  Q(1, 2) = -q.z();
  Q(1, 3) = q.y();
  Q(2, 3) = -q.x();
  // bottom left
  Q(1, 0) = q.x();
  Q(2, 0) = q.y();
  Q(3, 0) = q.z();
  Q(2, 1) = q.z();
  Q(3, 1) = -q.y();
  Q(3, 2) = q.x();
  return Q;
}

/**
 * Convert rotation vector in so(3) to quaternion. Returns quaternion corresponding to the
 * rotation matrix `R`, where: R = exp([w]_x).
 *
 * `w` must be convertible to a 3-element vector.
 *
 * As `|w| -> 0`, we take the limit and use a small angle approximation.
 */
template <typename Derived>
Quaternion<ScalarType<Derived>> QuaternionExp(const Eigen::MatrixBase<Derived>& w_xpr) {
  using Scalar = ScalarType<Derived>;
  constexpr Scalar kZeroTol = static_cast<Scalar>(1.0e-6);
  const Scalar angle = w_xpr.norm();
  // Fill out the quaternion.
  Eigen::Quaternion<Scalar> q;
  q.w() = std::cos(angle / 2);
  if (angle < kZeroTol) {
    q.vec() = w_xpr / 2;
    q.normalize();
  } else {
    const Scalar sinc_ha_2 = std::sin(angle / 2) / angle;
    q.vec() = w_xpr * sinc_ha_2;
  }
  return q;
}

/**
 * The derivative of QuaternionExp.
 *
 * Returns the 4x3 jacobian of the quaternion elements [w, x, y, z] wrt the
 * rodrigues parameters `w` (omega).
 *
 * As `|w| -> 0`, we take the limit and use a small angle approximation.
 */
template <typename Derived>
Matrix<ScalarType<Derived>, 4, 3> QuaternionExpJacobian(const Eigen::MatrixBase<Derived>& w_xpr) {
  using Scalar = ScalarType<Derived>;
  constexpr Scalar kZeroTol = static_cast<Scalar>(1.0e-6);
  const Vector<Scalar, 3> w = w_xpr.eval();
  const Scalar angle = w.norm();
  // Fill out the quaternion.
  Eigen::Quaternion<Scalar> q;
  Eigen::Matrix<Scalar, 4, 3> q_D_w;
  q.w() = std::cos(angle / 2);
  if (angle < kZeroTol) {
    // Small angle approx: lim sin(x)/x -> 1
    q.vec() = w / 2;
    q.normalize();
    // d(q.w) / d(w) = -sin(theta / 2) * (1 / 2) * (w^T / theta) = -[q.x, q.y, q.z]^T
    q_D_w.template topRows<1>() = -q.vec().transpose() / 2;
    q_D_w.template bottomRows<3>().setIdentity();
    q_D_w.template bottomRows<3>().diagonal() *= static_cast<Scalar>(0.5);
  } else {
    const Scalar sinc_ha_2 = std::sin(angle / 2) / angle;
    q.vec() = w * sinc_ha_2;
    // Fill out derivtive part. First row is same in both cases (small/large).
    const Vector<Scalar, 3> w_hat = w / angle;
    q_D_w.template topRows<1>() = -q.vec().transpose() / 2;
    q_D_w.template bottomRows<3>().setIdentity();
    q_D_w.template bottomRows<3>().diagonal() *= sinc_ha_2;
    q_D_w.template bottomRows<3>().noalias() +=
        w_hat * (w_hat.transpose() * (q.w() / 2 - sinc_ha_2));
  }
  return q_D_w;
}

/**
 * Convert a rotation type (either Matrix3 or Quaternion) to rodrigues vector.
 * This is equivalent to w = log(R) where `R = exp([w]_x)`.
 *
 * Strictly speaking, `log(R)` should return a skew-symmetric matrix. We assume the operator
 * also takes the off-diagonal elements, in order to recover the rotation vector.
 *
 * Note that because the returned angle is constrained to the the interval [0, pi], the
 * relationship above only holds for `w` that satisfy 0 <= |w| <= pi.
 */
template <typename RotationType>
Vector<typename RotationType::Scalar, 3> RotationLog(const RotationType& rot) {
  using Scalar = typename RotationType::Scalar;
  const Eigen::AngleAxis<Scalar> angle_axis(rot);
  return angle_axis.angle() * angle_axis.axis();
}

/**
 * Compute the jacobian of SO(3), for a given so(3)/rodrigues rotation vector.
 *
 * This is the jacobian of:
 *
 *   log(exp(w)^-1 * exp(w + dw)) with respect to dw, linearized about dw = 0
 *
 * It is also the inverse of the result of SO3JacobianInverse(), evaluated
 * analytically. You can obtain that relation as follows:
 *
 *    exp(v + dv) = exp(w) * exp(dw)
 *
 * Then:
 *
 *    v + dv = log[exp(w) * exp(dw)]
 *
 * Such that:
 *
 *    J = dv/dw = d(log[exp(w) * exp(dw)]) / dw
 *
 * Then dw/dv = J^-1 (which is invertible). J^-1 converts the additive perturbation of `dv`
 * to the group perturbation of `dw`(to first order):
 *
 *   exp(w + dw) ~= exp(w) * exp(dw)  (Note: Approximate, valid only for small dw).
 *
 * You might care about this derivative, for example, if you were integrating gyroscope data:
 *
 *    world_R_body[k+1] * exp(dR) = world_R_body[k] * exp(gyro + dGyro)
 *
 * The perturbation of the gyroscope data (dGyro) is additive, and we must propagate it onto
 * the body-frame tangent space of our integrated rotation, dR.
 *
 * In addition, if you were to execute the exponential map of SE(3):
 *
 *   exp([w_x, u]) = [exp(w_x) V(w) * u]
 *
 * The matrix V(w) is just SO3Jacobian(-w). In this context it is sometimes referred to
 * as the "left jacobian of SO(3)". Note that argument `w` was negated to get this relationship,
 * so the two jacobians are not equivalent.
 *
 * You can see this in more detail in:
 *
 * "Associating Uncertainty With Three-Dimensional Poses for Use in Estimation Problems"
 *   Tim. Barfoot and Paul Furgale, 2014
 */
template <typename Derived>
Matrix<ScalarType<Derived>, 3, 3> SO3Jacobian(const Eigen::MatrixBase<Derived>& w) {
  using Scalar = ScalarType<Derived>;
  const Scalar theta = w.norm();
  const Scalar theta2 = theta * theta;
  Matrix<Scalar, 3, 3> J = Matrix<Scalar, 3, 3>::Identity();
  if (theta < static_cast<Scalar>(1.0e-6)) {
    J.noalias() -= Skew3(w * static_cast<Scalar>(0.5));
    J.noalias() += w * w.transpose() * (1 / static_cast<Scalar>(6));
  } else {
    const Scalar sinc_theta = std::sin(theta) / theta;
    J.diagonal() *= sinc_theta;
    J.noalias() -= Skew3(w * (1 - std::cos(theta)) / theta2);
    J.noalias() += w * (w.transpose() * (1 - sinc_theta) / theta2);
  }
  return J;
}

/*
 * Computes the 3x3 Jacobian of:
 *
 *   v = log(exp([w]_x) * exp(dw)) with respect to `dw`, linearizing about dw = 0.
 *
 * You can get this expression fairly easily by evaluating the expression above via the
 * chain rule:
 *
 *   dlog(x)/dx|x=exp(w) * d(A*B)/dB|A=exp(w) * dexp(dw)/ddw|dw=0
 *
 * Where dlog(x)/dx is the 3x9 derivative of rodrigues params wrt matrix elements. You can
 * find it defined in:
 *
 *  "A tutorial on SE(3) transformation parameterization and on-manifold optimization"
 *    JL Blano, 2010 (Chapter 10)
 *
 * d(A*B)/dB is the Kronecker product: I3 \kron A = I3 \kron exp(w)
 * dexp(dw)/ddw evaluated at zero is just the generators of so3: [[-i]_x; [-j]_x; [-k]_x]
 *
 * Note that we are not taking the derivative with respect to the argument, but with respect to
 * d_w, and the derivative is always evaluated about zero.
 *
 * This is the analytical inverse of SO3Jacobian.
 */
template <typename Derived>
Matrix<ScalarType<Derived>, 3, 3> SO3JacobianInverse(const Eigen::MatrixBase<Derived>& w_xpr) {
  using Scalar = ScalarType<Derived>;
  const Matrix<Scalar, 3, 1> w = w_xpr.eval();
  const Scalar theta = w.norm();
  Eigen::Matrix<Scalar, 3, 3> J = Eigen::Matrix<Scalar, 3, 3>::Zero();
  if (theta < static_cast<Scalar>(1.0e-6)) {
    // small angle approximation
    J.diagonal().setConstant(1);
    J.noalias() += Skew3(w * 0.5);
    J.noalias() += w * w.transpose() * static_cast<Scalar>(1) / static_cast<Scalar>(12);
  } else {
    const Scalar cos = std::cos(theta);
    const Scalar sin = std::sin(theta);
    J.diagonal().setConstant(theta * (cos + 1) / (2 * sin));
    J.noalias() += Skew3(w * 0.5);
    J.noalias() += w * w.transpose() * -(1 + cos - 2 * sin / theta) / (2 * theta * sin);
  }
  return J;
}

/**
 * Compute the derivative of: `f(dw) = R * exp([dw]_x) * p` with respect to `dw`, linearizing
 * about dw = 0. Note this is only the derivative wrt the tangent-space of R.
 */
template <typename Scalar>
Matrix<Scalar, 3, 3> RotateVectorSO3TangentJacobian(const Eigen::Quaternion<Scalar>& R,
                                                    const Vector<Scalar, 3>& p) {
  return R * Skew3(-p);
}

/**
 * Functional struct for converting euler angles to SO(3), w/ derivatives.
 *
 * The rotation itself is stored in a quaternion, but the derivative is the tangent space
 * of SO(3) on the right wrt the three euler angles.
 *
 * Thus we are computing the rotation:
 *
 *    R_out = R_z * R_y * R_z (NOTE: Z in the left frame, X in the right)
 *
 * And the Jacobian is compute from:
 *
 *    R_out * exp(dw) = R_z(z + dz) * R_y(y + dy) * R_x(x + dx)
 *
 *    J = dw/d[dx,dy,dz]) (NOTE: Jacobian ordered: x, y, z)
 *
 * We use a struct in order to return both the rotation and the derivative.
 */
template <typename Scalar>
struct SO3FromEulerAngles_ {
  // Helper method to construct quaternion.
  template <typename BasisVectorType>
  static Quaternion<Scalar> QuatFromHalfAnglesAndUnitVector(const Scalar cos_half,
                                                            const Scalar sin_half,
                                                            const BasisVectorType& basis_vector) {
    // We don't need to deal w/ small angle here as the basis vector is normalized.
    Quaternion<Scalar> q;
    q.w() = cos_half;
    q.vec() = basis_vector * sin_half;
    return q;
  }

  // Construct from Vector, ordered [x, y, z].
  SO3FromEulerAngles_(const Vector<Scalar, 3>& xyz)
      : rotation_D_angles(Matrix<Scalar, 3, 3>::Zero()) {
    // Compute sin and cosines of the half angles
    const Scalar c_hx = std::cos(xyz.x() / 2);
    const Scalar s_hx = std::sin(xyz.x() / 2);
    const Scalar c_hy = std::cos(xyz.y() / 2);
    const Scalar s_hy = std::sin(xyz.y() / 2);
    const Scalar c_hz = std::cos(xyz.z() / 2);
    const Scalar s_hz = std::sin(xyz.z() / 2);
    // Fill out the quaternion itself.
    q = QuatFromHalfAnglesAndUnitVector(c_hz, s_hz, Vector<Scalar, 3>::UnitZ()) *
        QuatFromHalfAnglesAndUnitVector(c_hy, s_hy, Vector<Scalar, 3>::UnitY()) *
        QuatFromHalfAnglesAndUnitVector(c_hx, s_hx, Vector<Scalar, 3>::UnitX());
    // Cosines/sines we need for the derivatives.
    const Scalar cx = 2 * c_hx * c_hx - 1;
    const Scalar sx = 2 * c_hx * s_hx;
    const Scalar cy = 2 * c_hy * c_hy - 1;
    const Scalar sy = 2 * c_hy * s_hy;
    // dw/dx is simple, as x is in the right-side frame already, it is just [0, 0, 1]
    rotation_D_angles(0, 0) = 1;
    // dw/dy is a little more complex, we must multiply by the adjoint of R_x.
    // dw/dy = R_x^T * j_hat
    rotation_D_angles(1, 1) = cx;
    rotation_D_angles(2, 1) = -sx;
    // dw/dz is the hardest, since R_z is the left-most rotation.
    // We must account for the X and Y rotations:
    // dw/dz = R_x^T * R_y^T * k_hat
    rotation_D_angles(0, 2) = -sy;
    rotation_D_angles(1, 2) = sx * cy;
    rotation_D_angles(2, 2) = cx * cy;
  }

  // Rotation composed in order Rz * Ry * Rx
  Quaternion<Scalar> q;

  // Derivative of the right tangent-space of SO(3) wrt the input euler angles.
  Matrix<Scalar, 3, 3> rotation_D_angles;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Helper for calling the method above that deduces the template argument.
template <typename Derived>
SO3FromEulerAngles_<ScalarType<Derived>> SO3FromEulerAngles(const Eigen::MatrixBase<Derived>& xyz) {
  return SO3FromEulerAngles_<ScalarType<Derived>>(xyz);
}

/**
 * Derivative of the exponential map: so(3) -> SO(3).
 *
 * Returns the 9x3 jacobian of the elements of the 3x3 rotation matrix `R = exp([w]_x)` with
 * respect to the 3-element rodrigues vector `w`.
 *
 * This is the derivative `dvec(exp([w]_x)) / dw` where `vec` unpacks a matrix in
 * column order.
 *
 * This is the matrix version of QuaternionExpJacobian.
 *
 * TODO(gareth): It is possible this has a simpler form. I went as far as condensing it to
 * terms proportional to 1 / theta^-4 and stopped.
 */
template <typename Scalar>
Matrix<Scalar, 9, 3> SO3ExpMatrixJacobian(const Vector<Scalar, 3>& w) {
  const Scalar theta2 = w.squaredNorm();
  const Scalar theta = std::sqrt(theta2);
  const Scalar sin_theta = std::sin(theta);
  const Scalar cos_theta = std::cos(theta);

  // The result is 9x3, break into three vertical blocks of 3x3.
  // These correspond to the 3 columns of the output matrix.
  Matrix<Scalar, 9, 3> result = Matrix<Scalar, 9, 3>::Zero();
  auto top_block = result.template topRows<3>();
  auto middle_block = result.template block<3, 3>(3, 0);
  auto bottom_block = result.template bottomRows<3>();

  // Expression for unit vectors.
  static const auto i_hat = Vector<Scalar, 3>::UnitX();
  static const auto j_hat = Vector<Scalar, 3>::UnitY();
  static const auto k_hat = Vector<Scalar, 3>::UnitZ();
  static const auto I3 = Matrix<Scalar, 3, 3>::Identity();

  // Expression for w * w^T (a self-adjoint 3x3 matrix).
  const auto w_outer_prod = w * w.transpose();

  // Experssion for product of skew(w) * skew(w)
  const Matrix<Scalar, 3, 3> skew_w_sqr = w_outer_prod - I3 * theta2;

  if (theta2 < static_cast<Scalar>(1.0e-12)) {
    // take the limit near zero
    top_block(1, 2) = 1;
    top_block(2, 1) = -1;
    middle_block(0, 2) = -1;
    middle_block(2, 0) = 1;
    bottom_block(0, 1) = 1;
    bottom_block(1, 0) = -1;
  } else {
    // theta^-1 term, this is just [ [i]_x, [j]_x, [k]_x ] * -sin_theta / theta
    const Scalar sinc_theta = sin_theta / theta;
    top_block(1, 2) = sinc_theta;
    top_block(2, 1) = -sinc_theta;
    middle_block(0, 2) = -sinc_theta;
    middle_block(2, 0) = sinc_theta;
    bottom_block(0, 1) = sinc_theta;
    bottom_block(1, 0) = -sinc_theta;

    // The asymmetric part of the theta^-2 term.
    const Scalar asym_theta2_coeff = (1 - cos_theta) / theta2;
    top_block +=
        (I3 * w.x() + w * i_hat.transpose() - 2 * i_hat * w.transpose()) * asym_theta2_coeff;
    middle_block +=
        (I3 * w.y() + w * j_hat.transpose() - 2 * j_hat * w.transpose()) * asym_theta2_coeff;
    bottom_block +=
        (I3 * w.z() + w * k_hat.transpose() - 2 * k_hat * w.transpose()) * asym_theta2_coeff;

    // Add contribution of omega outer product.
    // Updating this way is faster than using UnitX().cross() or skew3.
    const Scalar outer_prod_coeff = (sinc_theta - cos_theta) / theta2;
    XHatMul(w_outer_prod * outer_prod_coeff, top_block);
    YHatMul(w_outer_prod * outer_prod_coeff, middle_block);
    ZHatMul(w_outer_prod * outer_prod_coeff, bottom_block);

    // theta^-3 and theta^-4 terms.
    // The map executes the equivalent of `vec([w]_x * [w]_x)` without copying.
    const Eigen::Map<const Vector<Scalar, 9>, Eigen::Aligned> vec_w_sqr(skew_w_sqr.data());
    const Scalar vec_skew_w_sqr_coeff =
        sinc_theta / theta2 - 2 * (1 - cos_theta) / (theta2 * theta2);
    result.noalias() += vec_w_sqr * (w.transpose() * vec_skew_w_sqr_coeff);
  }
  return result;
}

/**
 * Computes the 3x3 Jacobian of `v = log(R * exp([w]_x))` with respect to `w`.
 *
 * `R` is represented as a quaternion. `v` is the resulting rodrigues parameters
 * corresponding to the combined rotation `R * exp([w]_x)`.
 *
 * Note that this is distinct from SO3JacobianInverse, because in this context
 * we are taking the derivative wrt to the argument `w`, and this need not be
 * evaluated about zero.
 *
 * You can obtain this derivative by applying the chain rule to the expression for `v`.
 *
 * TODO(gareth): The name of this method is a bit muddled, but I'm not
 * sure what else to call it.
 */
template <typename Scalar>
Matrix<Scalar, 3, 3> SO3LogMulExpJacobian(const Quaternion<Scalar>& R, const Vector<Scalar, 3>& w) {
  const Quaternion<Scalar> exp_w = QuaternionExp(w);
  const Matrix<Scalar, 3, 3> B = (R * exp_w).matrix();

  // Derivative of exponential map.
  const Matrix<Scalar, 9, 3> exp_w_D_w = SO3ExpMatrixJacobian(w);
  const auto exp_w_D_w_top = exp_w_D_w.template topRows<3>();
  const auto exp_w_D_w_middle = exp_w_D_w.template block<3, 3>(3, 0);
  const auto exp_w_D_w_bottom = exp_w_D_w.template bottomRows<3>();

  // Recover the vector v' = f(B - B^T) where f is the inverse of the skew operator.
  // We define v' (v_prime) as follows: v = theta / (2 * sin(theta)) * v'
  // clang-format off
  const Matrix<Scalar, 3, 1> v_prime{
    -B(1, 2) + B(2, 1),
     B(0, 2) - B(2, 0),
    -B(0, 1) + B(1, 0)
  };
  // clang-format on

  // Derivative of v_prime wrt omega (need to make it a block for hat muls)
  Matrix<Scalar, 3, 3> v_prime_D_w = Matrix<Scalar, 3, 3>::Zero();
  auto v_prime_D_w_block = v_prime_D_w.template topLeftCorner<3, 3>();
  XHatMul(R * exp_w_D_w_top, v_prime_D_w_block);
  YHatMul(R * exp_w_D_w_middle, v_prime_D_w_block);
  ZHatMul(R * exp_w_D_w_bottom, v_prime_D_w_block);

  // Delegate to eigen for angle-axis conversion (we do the derivs ourselves).
  const Eigen::AngleAxis<Scalar> aa(B);
  const Scalar& theta = aa.angle();

  if (theta < static_cast<Scalar>(1.0e-6)) {
    // zero angle case, take the limit
    return v_prime_D_w * static_cast<Scalar>(0.5);
  }

  const Scalar cos_theta = std::cos(theta);
  const Scalar sin_theta = std::sin(theta);

  // Derivative of the theta angle wrt the trace of B.
  const Scalar theta_D_trace = -1 / (std::sqrt(1 - cos_theta * cos_theta) * 2);

  // Compute the normalization factor
  const Scalar norm_factor = theta / (2 * sin_theta);
  const Scalar norm_factor_D_theta = (-theta * cos_theta + sin_theta) / (2 * sin_theta * sin_theta);
  const Scalar norm_factor_D_trace = norm_factor_D_theta * theta_D_trace;

  // Derivative of trace wrt omega.
  const Matrix<Scalar, 1, 3> trace_D_w =
      (R * exp_w_D_w_top).row(0) + (R * exp_w_D_w_middle).row(1) + (R * exp_w_D_w_bottom).row(2);

  // Return the full derivative.
  return v_prime * (trace_D_w * norm_factor_D_trace) + (norm_factor * v_prime_D_w);
}

}  // namespace math
