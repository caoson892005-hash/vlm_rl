#include "social_navigation/social_layer.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_set>
#include <vector>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace social_navigation
{

void SocialLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("SocialLayer cannot lock lifecycle node");
  }
  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("people_topic", rclcpp::ParameterValue(std::string("/people")));
  declareParameter("groups_topic", rclcpp::ParameterValue(std::string("/people_groups")));
  declareParameter("individual_o_radius", rclcpp::ParameterValue(0.45));
  declareParameter("individual_p_front_radius", rclcpp::ParameterValue(1.1));
  declareParameter("individual_p_side_radius", rclcpp::ParameterValue(0.8));
  declareParameter("individual_p_rear_radius", rclcpp::ParameterValue(0.7));
  declareParameter("individual_r_front_radius", rclcpp::ParameterValue(1.5));
  declareParameter("individual_r_side_radius", rclcpp::ParameterValue(1.0));
  declareParameter("individual_r_rear_radius", rclcpp::ParameterValue(0.75));
  declareParameter("prediction_time", rclcpp::ParameterValue(1.5));
  declareParameter("moving_prediction_time", rclcpp::ParameterValue(3.0));
  declareParameter("moving_speed_threshold", rclcpp::ParameterValue(0.05));
  declareParameter("overlap_escape_clearance", rclcpp::ParameterValue(0.55));
  declareParameter("prediction_steps", rclcpp::ParameterValue(4));
  declareParameter("o_space_cost", rclcpp::ParameterValue(254));
  declareParameter("p_space_cost", rclcpp::ParameterValue(250));
  declareParameter("group_p_space_cost", rclcpp::ParameterValue(253));
  declareParameter("group_escape_cost", rclcpp::ParameterValue(200));
  declareParameter("group_escape_corridor_half_width", rclcpp::ParameterValue(0.55));
  declareParameter("r_space_cost", rclcpp::ParameterValue(100));
  declareParameter("moving_p_space_cost", rclcpp::ParameterValue(252));
  declareParameter("moving_r_space_cost", rclcpp::ParameterValue(140));
  declareParameter("data_timeout", rclcpp::ParameterValue(1.0));

  std::string people_topic;
  std::string groups_topic;
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".people_topic", people_topic);
  node->get_parameter(name_ + ".groups_topic", groups_topic);
  node->get_parameter(name_ + ".individual_o_radius", o_radius_);
  node->get_parameter(name_ + ".individual_p_front_radius", p_front_radius_);
  node->get_parameter(name_ + ".individual_p_side_radius", p_side_radius_);
  node->get_parameter(name_ + ".individual_p_rear_radius", p_rear_radius_);
  node->get_parameter(name_ + ".individual_r_front_radius", r_front_radius_);
  node->get_parameter(name_ + ".individual_r_side_radius", r_side_radius_);
  node->get_parameter(name_ + ".individual_r_rear_radius", r_rear_radius_);
  node->get_parameter(name_ + ".prediction_time", prediction_time_);
  node->get_parameter(name_ + ".moving_prediction_time", moving_prediction_time_);
  node->get_parameter(name_ + ".moving_speed_threshold", moving_speed_threshold_);
  node->get_parameter(name_ + ".overlap_escape_clearance", overlap_escape_clearance_);
  node->get_parameter(name_ + ".prediction_steps", prediction_steps_);
  node->get_parameter(name_ + ".o_space_cost", o_cost_);
  node->get_parameter(name_ + ".p_space_cost", p_cost_);
  node->get_parameter(name_ + ".group_p_space_cost", group_p_cost_);
  node->get_parameter(name_ + ".group_escape_cost", group_escape_cost_);
  node->get_parameter(
    name_ + ".group_escape_corridor_half_width", group_escape_corridor_half_width_);
  node->get_parameter(name_ + ".r_space_cost", r_cost_);
  node->get_parameter(name_ + ".moving_p_space_cost", moving_p_cost_);
  node->get_parameter(name_ + ".moving_r_space_cost", moving_r_cost_);
  node->get_parameter(name_ + ".data_timeout", data_timeout_);

  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_;
  people_sub_ = node->create_subscription<social_perception::msg::People>(
    people_topic, rclcpp::QoS(10),
    std::bind(&SocialLayer::peopleCallback, this, std::placeholders::_1), options);
  groups_sub_ = node->create_subscription<social_perception::msg::Groups>(
    groups_topic, rclcpp::QoS(10),
    std::bind(&SocialLayer::groupsCallback, this, std::placeholders::_1), options);
  current_ = true;
  matchSize();
  RCLCPP_INFO(logger_, "SocialLayer listening on %s and %s", people_topic.c_str(), groups_topic.c_str());
}

void SocialLayer::peopleCallback(
  const social_perception::msg::People::SharedPtr message)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  people_ = *message;
  people_received_at_ = clock_->now();
  have_people_message_ = true;
}

void SocialLayer::groupsCallback(
  const social_perception::msg::Groups::SharedPtr message)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  groups_ = *message;
  groups_received_at_ = clock_->now();
  have_groups_message_ = true;
}

void SocialLayer::reset()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  people_.people.clear();
  groups_.groups.clear();
  have_people_message_ = false;
  have_groups_message_ = false;
  group_escape_states_.clear();
  have_last_bounds_ = false;
  current_ = true;
}

void SocialLayer::updateBounds(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {return;}
  // Nav2 provides this pose in the costmap global frame. Keep it so an
  // already-overlapping person cannot surround the robot with an inescapable
  // lethal ring.
  robot_x_ = robot_x;
  robot_y_ = robot_y;
  robot_yaw_ = robot_yaw;
  have_robot_pose_ = true;
  social_perception::msg::People people;
  social_perception::msg::Groups groups;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    people = people_;
    groups = groups_;
    const auto now = clock_->now();
    if (!have_people_message_ || (now - people_received_at_).seconds() > data_timeout_) {
      people.people.clear();
    }
    if (!have_groups_message_ || (now - groups_received_at_).seconds() > data_timeout_) {
      groups.groups.clear();
    }
  }

  if (have_last_bounds_) {
    *min_x = std::min(*min_x, last_min_x_);
    *min_y = std::min(*min_y, last_min_y_);
    *max_x = std::max(*max_x, last_max_x_);
    *max_y = std::max(*max_y, last_max_y_);
  }

  double new_min_x = std::numeric_limits<double>::max();
  double new_min_y = std::numeric_limits<double>::max();
  double new_max_x = std::numeric_limits<double>::lowest();
  double new_max_y = std::numeric_limits<double>::lowest();
  bool found = false;
  const std::string target = layered_costmap_->getGlobalFrameID();

  auto include_point = [&](double x, double y, double radius) {
      new_min_x = std::min(new_min_x, x - radius);
      new_min_y = std::min(new_min_y, y - radius);
      new_max_x = std::max(new_max_x, x + radius);
      new_max_y = std::max(new_max_y, y + radius);
      found = true;
    };

  try {
    for (const auto & person : people.people) {
      geometry_msgs::msg::PoseStamped input;
      input.header = people.header;
      input.pose = person.pose;
      const auto output = tf_->transform(input, target, tf2::durationFromSec(0.1));
      geometry_msgs::msg::Vector3Stamped velocity_input;
      velocity_input.header = people.header;
      velocity_input.vector = person.velocity.linear;
      const auto velocity = tf_->transform(
        velocity_input, target, tf2::durationFromSec(0.1));
      include_point(output.pose.position.x, output.pose.position.y,
        std::max(r_front_radius_, r_side_radius_));
      const double speed = std::hypot(velocity.vector.x, velocity.vector.y);
      if (speed >= moving_speed_threshold_) {
        include_point(
          output.pose.position.x + velocity.vector.x * moving_prediction_time_,
          output.pose.position.y + velocity.vector.y * moving_prediction_time_,
          std::max(r_front_radius_, r_side_radius_));
      }
    }
    for (const auto & group : groups.groups) {
      geometry_msgs::msg::PointStamped input;
      input.header = groups.header;
      input.point = group.center;
      const auto output = tf_->transform(input, target, tf2::durationFromSec(0.1));
      include_point(output.point.x, output.point.y, group.r_radius);
    }
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "SocialLayer transform failed: %s", error.what());
  }

  if (found) {
    last_min_x_ = new_min_x;
    last_min_y_ = new_min_y;
    last_max_x_ = new_max_x;
    last_max_y_ = new_max_y;
    have_last_bounds_ = true;
    *min_x = std::min(*min_x, new_min_x);
    *min_y = std::min(*min_y, new_min_y);
    *max_x = std::max(*max_x, new_max_x);
    *max_y = std::max(*max_y, new_max_y);
  } else {
    have_last_bounds_ = false;
  }
}

void SocialLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {return;}
  social_perception::msg::People people;
  social_perception::msg::Groups groups;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    people = people_;
    groups = groups_;
    const auto now = clock_->now();
    if (!have_people_message_ || (now - people_received_at_).seconds() > data_timeout_) {
      people.people.clear();
    }
    if (!have_groups_message_ || (now - groups_received_at_).seconds() > data_timeout_) {
      groups.groups.clear();
    }
  }

  struct PersonInMap {double x; double y; double vx; double vy; double speed;};
  struct GroupInMap
  {
    std::string id;
    double x;
    double y;
    double o;
    double p;
    double r;
  };
  std::vector<PersonInMap> mapped_people;
  std::vector<GroupInMap> mapped_groups;
  const std::string target = layered_costmap_->getGlobalFrameID();
  try {
    for (const auto & person : people.people) {
      geometry_msgs::msg::PoseStamped input;
      input.header = people.header;
      input.pose = person.pose;
      const auto output = tf_->transform(input, target, tf2::durationFromSec(0.1));
      geometry_msgs::msg::Vector3Stamped velocity_input;
      velocity_input.header = people.header;
      velocity_input.vector = person.velocity.linear;
      const auto velocity = tf_->transform(
        velocity_input, target, tf2::durationFromSec(0.1));
      mapped_people.push_back({output.pose.position.x, output.pose.position.y,
        velocity.vector.x, velocity.vector.y,
        std::hypot(velocity.vector.x, velocity.vector.y)});
    }
    for (const auto & group : groups.groups) {
      geometry_msgs::msg::PointStamped input;
      input.header = groups.header;
      input.point = group.center;
      const auto output = tf_->transform(input, target, tf2::durationFromSec(0.1));
      mapped_groups.push_back({group.id, output.point.x, output.point.y,
        group.o_radius, group.p_radius, group.r_radius});
    }
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "SocialLayer transform failed: %s", error.what());
    return;
  }

  const int begin_i = std::max(0, min_i);
  const int begin_j = std::max(0, min_j);
  const int end_i = std::min(static_cast<int>(master.getSizeInCellsX()), max_i);
  const int end_j = std::min(static_cast<int>(master.getSizeInCellsY()), max_j);

  // Latch the escape direction when a late group first surrounds the robot.
  // Without this state, the centre fallback would rotate together with the
  // robot while DWB turns around, leaving the exit perpetually behind it.
  std::unordered_set<std::string> current_group_ids;
  for (const auto & group : mapped_groups) {
    current_group_ids.insert(group.id);
    const double robot_distance = have_robot_pose_ ?
      std::hypot(robot_x_ - group.x, robot_y_ - group.y) :
      std::numeric_limits<double>::infinity();
    if (group.p > 0.0 && robot_distance <= group.p) {
      if (group_escape_states_.find(group.id) == group_escape_states_.end()) {
        GroupEscapeState state;
        if (robot_distance > 0.10) {
          state.direction_x = (robot_x_ - group.x) / robot_distance;
          state.direction_y = (robot_y_ - group.y) / robot_distance;
        } else {
          state.direction_x = -std::cos(robot_yaw_);
          state.direction_y = -std::sin(robot_yaw_);
        }
        group_escape_states_.emplace(group.id, state);
      }
    } else if (robot_distance > group.p + overlap_escape_clearance_) {
      // Keep the corridor until the complete footprint and inflation margin
      // have crossed the P-space boundary, then seal it behind the robot.
      group_escape_states_.erase(group.id);
    }
  }
  for (auto state = group_escape_states_.begin(); state != group_escape_states_.end();) {
    if (current_group_ids.find(state->first) == current_group_ids.end()) {
      state = group_escape_states_.erase(state);
    } else {
      ++state;
    }
  }

  resetMap(begin_i, begin_j, end_i, end_j);
  for (int j = begin_j; j < end_j; ++j) {
    for (int i = begin_i; i < end_i; ++i) {
      double wx, wy;
      master.mapToWorld(i, j, wx, wy);
      unsigned char social_cost = nav2_costmap_2d::FREE_SPACE;

      for (const auto & person : mapped_people) {
        const int steps = std::max(1, prediction_steps_);
        const bool moving = person.speed >= moving_speed_threshold_;
        const double horizon = moving ? moving_prediction_time_ : prediction_time_;
        // Body orientation is intentionally ignored. A person turning in place
        // must not rotate the costmap and invalidate the robot's path. Only a
        // real velocity vector gives moving people an anisotropic social space.
        const double motion_yaw = moving ? std::atan2(person.vy, person.vx) : 0.0;
        for (int step = 0; step <= steps; ++step) {
          const double predicted_time = horizon * step / steps;
          const double center_x = person.x + (moving ? person.vx * predicted_time : 0.0);
          const double center_y = person.y + (moving ? person.vy * predicted_time : 0.0);
          const double dx = wx - center_x;
          const double dy = wy - center_y;
          const double forward = std::cos(motion_yaw) * dx + std::sin(motion_yaw) * dy;
          const double side = -std::sin(motion_yaw) * dx + std::cos(motion_yaw) * dy;
          const double distance = std::hypot(dx, dy);
          const double p_long = moving ?
            (forward >= 0.0 ? p_front_radius_ : p_rear_radius_) : p_side_radius_;
          const double r_long = moving ?
            (forward >= 0.0 ? r_front_radius_ : r_rear_radius_) : r_side_radius_;
          const double p_norm = (forward * forward) / (p_long * p_long) +
            (side * side) / (p_side_radius_ * p_side_radius_);
          const double r_norm = (forward * forward) / (r_long * r_long) +
            (side * side) / (r_side_radius_ * r_side_radius_);
          int cost = 0;
          const double robot_distance = have_robot_pose_ ?
            std::hypot(robot_x_ - center_x, robot_y_ - center_y) :
            std::numeric_limits<double>::infinity();
          const bool escaping_overlap = robot_distance <= o_radius_ &&
            distance >= std::max(0.0, robot_distance - overlap_escape_clearance_);
          if (distance <= o_radius_) {
            // Keep the inward side lethal, but downgrade the outward side to
            // P-space so DWB can produce a trajectory that separates them.
            cost = escaping_overlap ? p_cost_ : o_cost_;
          }
          else if (p_norm <= 1.0) {cost = moving ? moving_p_cost_ : p_cost_;}
          else if (r_norm <= 1.0) {
            // R-space is traversable. Use a decreasing gradient instead of a
            // uniform plateau so a robot already inside it is encouraged to
            // keep moving toward the lower-cost outer edge.
            const double edge_factor = std::clamp(1.0 - std::sqrt(r_norm), 0.0, 1.0);
            const int maximum_r_cost = moving ? moving_r_cost_ : r_cost_;
            cost = static_cast<int>(std::round(maximum_r_cost * edge_factor));
          }
          social_cost = std::max(social_cost,
            static_cast<unsigned char>(std::clamp(cost, 0, 254)));
        }
      }

      for (const auto & group : mapped_groups) {
        const double distance = std::hypot(wx - group.x, wy - group.y);
        const auto escape_state = group_escape_states_.find(group.id);
        const bool escape_active = escape_state != group_escape_states_.end();

        // A confirmed conversation is not merely an expensive shortcut: its
        // P-space is impassable. If VLM reports it after the robot has already
        // entered, expose exactly one traversable corridor back toward the side
        // from which the robot approached. The O/P costs on every other side
        // remain collision costs, so the replanner cannot continue through the
        // people and call that an "escape".
        double escape_x = 0.0;
        double escape_y = 0.0;
        if (escape_active) {
          escape_x = escape_state->second.direction_x;
          escape_y = escape_state->second.direction_y;
        }
        const double from_robot_x = wx - robot_x_;
        const double from_robot_y = wy - robot_y_;
        const double escape_forward =
          from_robot_x * escape_x + from_robot_y * escape_y;
        const double escape_side = std::abs(
          -from_robot_x * escape_y + from_robot_y * escape_x);
        const bool in_escape_corridor = escape_active &&
          escape_forward >= -overlap_escape_clearance_ &&
          escape_side <= group_escape_corridor_half_width_;
        int cost = 0;
        if (distance <= group.o) {
          cost = in_escape_corridor ? group_escape_cost_ : o_cost_;
        }
        else if (distance <= group.p) {
          cost = in_escape_corridor ? group_escape_cost_ : group_p_cost_;
        }
        else if (distance <= group.r) {
          const double width = std::max(1e-6, group.r - group.p);
          const double edge_factor = std::clamp((group.r - distance) / width, 0.0, 1.0);
          cost = static_cast<int>(std::round(r_cost_ * edge_factor));
        }
        social_cost = std::max(social_cost,
          static_cast<unsigned char>(std::clamp(cost, 0, 254)));
      }
      if (social_cost > nav2_costmap_2d::FREE_SPACE) {setCost(i, j, social_cost);}
    }
  }
  updateWithMax(master, begin_i, begin_j, end_i, end_j);
}

}  // namespace social_navigation

PLUGINLIB_EXPORT_CLASS(social_navigation::SocialLayer, nav2_costmap_2d::Layer)
