#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace autoaim_research {

// Mean-field Gaussian/Gamma update for ONE Student-t measurement component.
// All iterations condition on the same prior and frozen measurement Jacobian;
// this is not repeated assimilation and is not Roth's joint-t KL filter.
// range_scale is the t scale squared, NOT its marginal variance.
inline double studentTRangePrecision(
    const Eigen::MatrixXd& prior, const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& measurement_scale, const Eigen::VectorXd& innovation,
    double nu, int iterations = 30) {
  if (!(nu > 2.0) || !std::isfinite(nu) || iterations < 1 ||
      H.rows() != 4 || innovation.size() != 4 ||
      !(measurement_scale(2, 2) > 0.0)) {
    throw std::invalid_argument("invalid Student-t range update");
  }
  const Eigen::MatrixXd projected = H * prior * H.transpose();
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(prior.rows(), prior.cols());
  double precision = 1.0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    Eigen::MatrixXd effective = measurement_scale;
    effective(2, 2) /= precision;
    Eigen::LDLT<Eigen::MatrixXd> solve(projected + effective);
    if (solve.info() != Eigen::Success || !solve.isPositive())
      throw std::runtime_error("non-positive innovation covariance");
    const Eigen::MatrixXd gain = solve.solve(H * prior).transpose();
    const Eigen::VectorXd delta = gain * innovation;
    const Eigen::MatrixXd residual_map = identity - gain * H;
    const Eigen::MatrixXd posterior = residual_map * prior * residual_map.transpose()
        + gain * effective * gain.transpose();
    const double residual = innovation[2] - (H * delta)[2];
    const double expected_square = residual * residual
        + (H.row(2) * posterior * H.row(2).transpose())(0, 0);
    const double next = (nu + 1.0) /
        (nu + expected_square / measurement_scale(2, 2));
    if (!(next > 0.0) || !std::isfinite(next))
      throw std::runtime_error("invalid Student-t posterior precision");
    const bool converged = std::abs(next - precision) <= 1e-8 * (1.0 + precision);
    precision = next;
    if (converged) break;
  }
  return precision;
}

}  // namespace autoaim_research
