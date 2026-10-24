#pragma once

#include <chrono>
#include <functional>
#include <list>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "autoaim_research/tongji_target_with_association.hpp"

namespace autoaim_research {

struct TrackerConfig {
  int min_detect_count = 3;
  int max_temp_lost_count = 15;
  double max_observation_gap_s = 0.1;
  double initial_radius_m = 0.2;
  int armor_count = 4;
  // Multiplies Tongji's distance variance while preserving its incidence-angle
  // dependence. The production-equivalent default is exactly 1.0.
  double distance_measurement_variance_scale = 1.0;
  // Zero preserves the upstream update exactly. Positive values (>2) enable
  // the experimental scalar-range variational Student-t likelihood.
  double student_t_range_dof = 0.0;
  // Optional, observation-local empirical likelihood for research replay.
  // The empty default preserves the frozen 11D-EKF path.
  std::function<auto_aim::ResearchRangeLikelihood(const auto_aim::Armor&)> range_likelihood;
};

struct TrackerSnapshot {
  std::string state{"lost"};
  bool has_estimate = false;
  Eigen::VectorXd state_vector;
  // One entry per finite, image-center-sorted input armor. -1 means the
  // candidate did not match the active target class and was not consumed.
  std::vector<int> association_ids;
};

// Single-target research wrapper around the upstream 11-state Target EKF.
// It deliberately omits aiming, fire control and gimbal command generation.
class TongjiResearchTracker {
 public:
  explicit TongjiResearchTracker(TrackerConfig config = {});

  TrackerSnapshot update(std::list<auto_aim::Armor> armors,
                         std::uint64_t timestamp_ns);
  TrackerSnapshot updateAssociated(std::list<auto_aim::Armor> armors,
                                   std::uint64_t timestamp_ns,
                                   const std::vector<int>& association_ids);
  const std::string& state() const noexcept { return state_; }

 private:
  bool initialize(const std::list<auto_aim::Armor>& armors,
                  std::chrono::steady_clock::time_point timestamp);
  bool updateTarget(const std::list<auto_aim::Armor>& armors,
                    std::chrono::steady_clock::time_point timestamp,
                    const std::vector<int>* forced_association_ids,
                    std::vector<int>* applied_association_ids);
  TrackerSnapshot updateImpl(std::list<auto_aim::Armor> armors,
                             std::uint64_t timestamp_ns,
                             const std::vector<int>* forced_association_ids);
  void updateState(bool found);

  TrackerConfig config_;
  auto_aim::ResearchTarget target_;
  bool target_initialized_ = false;
  std::string state_{"lost"};
  int detect_count_ = 0;
  int temp_lost_count_ = 0;
  std::uint64_t last_timestamp_ns_ = 0;
};

}  // namespace autoaim_research
