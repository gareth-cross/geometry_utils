// Copyright 2020 Gareth Cross
#pragma once
#include "geometry_utils/matrix_types.hpp"

// Miscellaneous rotation utility functions.
namespace math {

/**
 * Compute basis that achieves the given Z-axis, and whose X and Y axes minimize the angular
 * distance to the provided X and Y vectors.
 *
 * `basis_z` should be a unit vector for this to work.
 *
 * Operates by first setting up a basis F = [bx, by, bz] where bz = `basis_z` and bx, by are
 * two other arbitrarily chosen vectors to create an orthonormal basis.
 *
 * Then we maximize: tx.dot(F * u) + ty.dot(F * v) where u = [cos(theta), sin(theta)] and v
 * is u rotated 90 degrees: v = [cos(theta + pi/2), sin(theta + pi/2)].
 *
 * In the case where basis_z is a reflection of the input basis z-axis (as determined by target_x
 * and target_y), we arbitrarily choose to maximize alignment with `target_x`, since maximizing both
 * is impossible.
 */
template <typename Scalar>
Matrix<Scalar, 3, 3> BasisFromZAxisWithMinAngularXY(const Vector<Scalar, 3>& basis_z,
                                                    const Vector<Scalar, 3>& target_x,
                                                    const Vector<Scalar, 3>& target_y) {
  // First pick two vectors orthonormal to basis_z. We do a check here that we are selecting
  // a vector that is not parallel or anti-parallel to basisZ:
  const Vector<Scalar, 3> basis_x = (std::abs(basis_z.z()) < static_cast<Scalar>(1 - 0.01f))
                                        ? basis_z.cross(Vector<Scalar, 3>::UnitZ()).normalized()
                                        : basis_z.cross(Vector<Scalar, 3>::UnitX()).normalized();
  const Vector<Scalar, 3> basis_y = basis_z.cross(basis_x);
  const Matrix<Scalar, 3, 2> F = (Matrix<Scalar, 3, 2>() << basis_x, basis_y).finished();

  // Project the target x/y into the plane we defined:
  const Vector<Scalar, 2> m = F.transpose() * target_x;
  const Vector<Scalar, 2> n = F.transpose() * target_y;

  // Compute angle theta that defines the rotation of the x-axis in the plane:
  // We do this by maximizing: m.dot(u) + n.dot(v)
  const Scalar c0 = m.x() + n.y();  //  n is rotated 90 degrees here
  const Scalar c1 = m.y() - n.x();
  const Scalar c_norm_squared = c0 * c0 + c1 * c1;
  const Scalar c_norm = std::sqrt(c_norm_squared);

  if (c_norm < static_cast<Scalar>(1.0e-12)) {
    // This case occurs when `m` and `n` are perpendicular in the chosen plane. This can happen
    // if the `target_x` and `target_y` already fall in a plane parallel to the one we defined.
    // In this case, take theta ~= 0. This is a choice, since we could maximize the alignment to
    // x or y, but not both.
    const auto final_basis_x = F.template leftCols<1>();
    return (Matrix<Scalar, 3, 3>() << final_basis_x, basis_z.cross(final_basis_x), basis_z)
        .finished();
  }

  // c_norm must be > 0 because both target_x and target_y cannot be parallel to basis_z
  const Scalar cos_theta = c0 / c_norm;
  const Scalar sin_theta = c1 / c_norm;
  const Vector<Scalar, 3> final_basis_x = F * Vector<Scalar, 2>{cos_theta, sin_theta};

  // Create final rotation matrix, using the fact that `final_basis_x` and `basis_z` are already
  // orthonormal.
  return (Matrix<Scalar, 3, 3>() << final_basis_x, basis_z.cross(final_basis_x), basis_z)
      .finished();
}

}  // namespace math
