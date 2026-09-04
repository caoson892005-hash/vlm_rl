#ifndef SOCIAL_NAVIGATION__SOCIAL_LAYER_HPP_
#define SOCIAL_NAVIGATION__SOCIAL_LAYER_HPP_

#include <mutex>
#include <string>
#include <unordered_map>

#include "nav2_costmap_2d/costmap_layer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "social_perception/msg/groups.hpp"
#include "social_perception/msg/people.hpp"

namespace social_navigation
{

class SocialLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  SocialLayer() = default;
  void onInitialize() override;
  void updateBounds(double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  bool isClearable() override {return true;}

private:
  struct GroupEscapeState
  {
    double direction_x;
    double direction_y;
  };

  void peopleCallback(const social_perception::msg::People::SharedPtr message);
  void groupsCallback(const social_perception::msg::Groups::SharedPtr message);

  rclcpp::Subscription<social_perception::msg::People>::SharedPtr people_sub_;
  rclcpp::Subscription<social_perception::msg::Groups>::SharedPtr groups_sub_;
  social_perception::msg::People people_;
  social_perception::msg::Groups groups_;
  rclcpp::Time people_received_at_{0, 0, RCL_ROS_TIME};
  rclcpp::Time groups_received_at_{0, 0, RCL_ROS_TIME};
  bool have_people_message_{false};
  bool have_groups_message_{false};
  std::mutex data_mutex_;
  double o_radius_{0.45};
  double p_front_radius_{1.1};
  double p_side_radius_{0.8};
  double p_rear_radius_{0.7};
  double r_front_radius_{1.5};
  double r_side_radius_{1.0};
  double r_rear_radius_{0.75};
  double prediction_time_{1.5};
  double moving_prediction_time_{3.0};
  double moving_speed_threshold_{0.05};
  double overlap_escape_clearance_{0.55};
  int prediction_steps_{4};
  double data_timeout_{1.0};
  int o_cost_{254};
  int p_cost_{250};
  int group_p_cost_{253};
  int group_escape_cost_{200};
  double group_escape_corridor_half_width_{0.55};
  int r_cost_{100};
  int moving_p_cost_{252};
  int moving_r_cost_{140};
  double last_min_x_{0.0};
  double last_min_y_{0.0};
  double last_max_x_{0.0};
  double last_max_y_{0.0};
  bool have_last_bounds_{false};
  double robot_x_{0.0};
  double robot_y_{0.0};
  double robot_yaw_{0.0};
  bool have_robot_pose_{false};
  std::unordered_map<std::string, GroupEscapeState> group_escape_states_;
};

}  // namespace social_navigation
#endif
