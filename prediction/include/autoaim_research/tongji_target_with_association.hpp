#ifndef AUTOAIM_RESEARCH__TONGJI_TARGET_WITH_ASSOCIATION_HPP
#define AUTOAIM_RESEARCH__TONGJI_TARGET_WITH_ASSOCIATION_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

struct ResearchRangeLikelihood
{
  double mean_error_m = 0.0;
  // Unset retains Tongji's original distance variance.
  std::optional<double> variance_m2;
};

class ResearchTarget
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  ResearchTarget() = default;
  ResearchTarget(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, double distance_variance_scale = 1.0,
    double student_t_range_dof = 0.0);
  ResearchTarget(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor);
  int association_id(const Armor & armor) const;
  void update_associated(const Armor & armor, int id);
  void update_associated_with_range_likelihood(
    const Armor & armor, int id, const ResearchRangeLikelihood & likelihood);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;
  double distance_variance_scale_ = 1.0;
  double student_t_range_dof_ = 0.0;

  void update_associated_impl(
    const Armor & armor, int id, const ResearchRangeLikelihood * likelihood);
  void update_ypda(
    const Armor & armor, int id, const ResearchRangeLikelihood * likelihood);  // yaw pitch distance angle

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTOAIM_RESEARCH__TONGJI_TARGET_WITH_ASSOCIATION_HPP
