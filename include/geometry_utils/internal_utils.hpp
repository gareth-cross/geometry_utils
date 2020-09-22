// Copyright 2020 Gareth Cross
#pragma once
#include "geometry_utils/matrix_types.hpp"

namespace math {

// Multiply by [i]_x on the left of a statically sized matrix.
// Writes the result into `out_expression`.
template <typename Derived, typename XprType, int BlockCols>
void XHatMul(const Eigen::MatrixBase<Derived>& rhs,
             Eigen::Block<XprType, 3, BlockCols, false> out_expression) {
  // Make sure the input was [3 x static].
  static_assert(3 == Eigen::MatrixBase<Derived>::RowsAtCompileTime, "");
  static_assert(BlockCols == Eigen::MatrixBase<Derived>::ColsAtCompileTime, "Cols should match");
  out_expression.row(1) -= rhs.row(2);
  out_expression.row(2) += rhs.row(1);
}

// Multiply by [j]_x on the left of a statically sized matrix.
// Writes the result into `out_expression`.
template <typename Derived, typename XprType, int BlockCols>
void YHatMul(const Eigen::MatrixBase<Derived>& rhs,
             Eigen::Block<XprType, 3, BlockCols, false> out_expression) {
  // Make sure the input was [3 x static].
  static_assert(3 == Eigen::MatrixBase<Derived>::RowsAtCompileTime, "");
  static_assert(BlockCols == Eigen::MatrixBase<Derived>::ColsAtCompileTime, "Cols should match");
  out_expression.row(0) += rhs.row(2);
  out_expression.row(2) -= rhs.row(0);
}

// Multiply by [k]_x on the left of a statically sized matrix.
// Writes the result into blo `out_expression`.
template <typename Derived, typename XprType, int BlockCols>
void ZHatMul(const Eigen::MatrixBase<Derived>& rhs,
             Eigen::Block<XprType, 3, BlockCols, false> out_expression) {
  // Make sure the input was [3 x static].
  static_assert(3 == Eigen::MatrixBase<Derived>::RowsAtCompileTime, "");
  static_assert(BlockCols == Eigen::MatrixBase<Derived>::ColsAtCompileTime, "Cols should match");
  out_expression.row(0) -= rhs.row(1);
  out_expression.row(1) += rhs.row(0);
}

}  // namespace math
