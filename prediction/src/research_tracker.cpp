#include "autoaim_research/research_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace autoaim_research {

TongjiResearchTracker::TongjiResearchTracker(TrackerConfig config)
    : config_(config) {
  if (config_.min_detect_count < 1 || config_.max_temp_lost_count < 0 ||
      config_.max_observation_gap_s <= 0.0 || config_.initial_radius_m <= 0.0 ||
      config_.armor_count < 3 ||
      !std::isfinite(config_.distance_measurement_variance_scale) ||
      config_.distance_measurement_variance_scale <= 0.0 ||
      !std::isfinite(config_.student_t_range_dof) ||
      (config_.student_t_range_dof != 0.0 && config_.student_t_range_dof <= 2.0)) {
    throw std::runtime_error("invalid research tracker configuration");
  }
}

TrackerSnapshot TongjiResearchTracker::update(
    std::list<auto_aim::Armor> armors, std::uint64_t timestamp_ns) {
  return updateImpl(std::move(armors), timestamp_ns, nullptr);
}

TrackerSnapshot TongjiResearchTracker::updateAssociated(
    std::list<auto_aim::Armor> armors, std::uint64_t timestamp_ns,
    const std::vector<int>& association_ids) {
  return updateImpl(std::move(armors), timestamp_ns, &association_ids);
}

TrackerSnapshot TongjiResearchTracker::updateImpl(
    std::list<auto_aim::Armor> armors, std::uint64_t timestamp_ns,
    const std::vector<int>* forced_association_ids) {
  if (timestamp_ns == 0 ||
      (last_timestamp_ns_ != 0 && timestamp_ns <= last_timestamp_ns_)) {
    state_ = "lost";
    target_initialized_ = false;
  }
  if (last_timestamp_ns_ != 0 && timestamp_ns > last_timestamp_ns_ &&
      static_cast<double>(timestamp_ns - last_timestamp_ns_) * 1e-9 >
          config_.max_observation_gap_s) {
    state_ = "lost";
    target_initialized_ = false;
  }
  last_timestamp_ns_ = timestamp_ns;

  armors.remove_if([](const auto_aim::Armor& armor) {
    return !armor.xyz_in_world.allFinite() || !armor.ypr_in_world.allFinite();
  });
  armors.sort([](const auto_aim::Armor& left, const auto_aim::Armor& right) {
    constexpr double center_x = 720.0;
    constexpr double center_y = 540.0;
    const double left_distance =
        std::hypot(left.center.x - center_x, left.center.y - center_y);
    const double right_distance =
        std::hypot(right.center.x - center_x, right.center.y - center_y);
    return left_distance < right_distance;
  });

  const auto timestamp = std::chrono::steady_clock::time_point{
      std::chrono::nanoseconds{timestamp_ns}};
  std::vector<int> applied_association_ids;
  const bool found = state_ == "lost" ? initialize(armors, timestamp)
                                         : updateTarget(armors, timestamp,
                                                        forced_association_ids,
                                                        &applied_association_ids);
  updateState(found);

  if (target_initialized_ && state_ != "lost" && target_.diverged()) {
    state_ = "lost";
    target_initialized_ = false;
  }

  TrackerSnapshot snapshot;
  snapshot.state = state_;
  snapshot.has_estimate = target_initialized_ && state_ != "lost";
  snapshot.association_ids = std::move(applied_association_ids);
  if (snapshot.has_estimate) snapshot.state_vector = target_.ekf_x();
  return snapshot;
}

bool TongjiResearchTracker::initialize(
    const std::list<auto_aim::Armor>& armors,
    std::chrono::steady_clock::time_point timestamp) {
  if (armors.empty()) return false;
  const auto& armor = armors.front();
  Eigen::VectorXd initial_covariance{
      {1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
  target_ = auto_aim::ResearchTarget(armor, timestamp,
                                     config_.initial_radius_m,
                                     config_.armor_count, initial_covariance,
                                     config_.distance_measurement_variance_scale,
                                     config_.student_t_range_dof);
  target_initialized_ = true;
  return true;
}

bool TongjiResearchTracker::updateTarget(
    const std::list<auto_aim::Armor>& armors,
    std::chrono::steady_clock::time_point timestamp,
    const std::vector<int>* forced_association_ids,
    std::vector<int>* applied_association_ids) {
  if (!target_initialized_) return false;
  if (forced_association_ids != nullptr &&
      forced_association_ids->size() != armors.size()) {
    throw std::runtime_error("forced association count differs from sorted armor count");
  }
  target_.predict(timestamp);
  bool found = false;
  std::size_t index = 0;
  applied_association_ids->reserve(armors.size());
  for (const auto& candidate : armors) {
    if (candidate.name != target_.name || candidate.type != target_.armor_type) {
      applied_association_ids->push_back(-1);
      ++index;
      continue;
    }
    auto armor = candidate;
    const int association_id = forced_association_ids == nullptr
        ? target_.association_id(armor)
        : (*forced_association_ids)[index];
    if (association_id < 0) {
      applied_association_ids->push_back(-1);
      ++index;
      continue;
    }
    if (config_.range_likelihood) {
      target_.update_associated_with_range_likelihood(
          armor, association_id, config_.range_likelihood(armor));
    } else {
      target_.update_associated(armor, association_id);
    }
    applied_association_ids->push_back(association_id);
    found = true;
    ++index;
  }
  return found;
}

void TongjiResearchTracker::updateState(bool found) {
  if (state_ == "lost") {
    if (!found) return;
    state_ = "detecting";
    detect_count_ = 1;
  } else if (state_ == "detecting") {
    if (!found) {
      state_ = "lost";
      target_initialized_ = false;
      detect_count_ = 0;
    } else if (++detect_count_ >= config_.min_detect_count) {
      state_ = "tracking";
    }
  } else if (state_ == "tracking") {
    if (!found) {
      state_ = "temp_lost";
      temp_lost_count_ = 1;
    }
  } else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
      temp_lost_count_ = 0;
    } else if (++temp_lost_count_ > config_.max_temp_lost_count) {
      state_ = "lost";
      target_initialized_ = false;
    }
  }
}

}  // namespace autoaim_research
