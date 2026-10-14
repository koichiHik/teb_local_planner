#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"
#include "teb_local_planner/homotopy_class_planner.h"
#include "teb_local_planner/robot_footprint_model.h"

namespace {

class RclcppFixture {
 public:
  RclcppFixture() { rclcpp::init(0, nullptr); }

  ~RclcppFixture() { rclcpp::shutdown(); }
};

RclcppFixture g_rclcpp_fixture;

class PlannedResultExportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    node_ = std::make_shared<nav2_util::LifecycleNode>("planned_result_export_test");

    cfg_.robot_model = std::make_shared<teb_local_planner::PointRobotFootprint>();
    cfg_.robot.base_max_vel_x = cfg_.robot.max_vel_x;
    cfg_.robot.base_max_vel_x_backwards = cfg_.robot.max_vel_x_backwards;
    cfg_.robot.base_max_vel_y = cfg_.robot.max_vel_y;
    cfg_.robot.base_max_vel_theta = cfg_.robot.max_vel_theta;

    cfg_.obstacles.include_dynamic_obstacles = false;
    cfg_.obstacles.include_costmap_obstacles = false;
    cfg_.optim.no_inner_iterations = 2;
    cfg_.optim.no_outer_iterations = 2;
    cfg_.optim.weight_viapoint = 0.0;

    cfg_.hcp.enable_homotopy_class_planning = true;
    cfg_.hcp.enable_multithreading = true;
    cfg_.hcp.simple_exploration = true;
    cfg_.hcp.max_number_classes = 1;
    cfg_.hcp.max_number_plans_in_current_class = 1;
    cfg_.hcp.delete_detours_backwards = false;

    planner_ = std::make_unique<teb_local_planner::HomotopyClassPlanner>(node_, cfg_, &obstacles_);
  }

  std::vector<geometry_msgs::msg::PoseStamped> makeStraightPlan() const {
    std::vector<geometry_msgs::msg::PoseStamped> plan;
    for (int i = 0; i <= 5; ++i) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "odom";
      pose.pose.position.x = 0.4 * static_cast<double>(i);
      pose.pose.orientation.w = 1.0;
      plan.push_back(pose);
    }
    return plan;
  }

  teb_local_planner::TebOptimalPlannerPtr planSuccessfully() {
    const auto plan = makeStraightPlan();
    EXPECT_TRUE(planner_->plan(plan));
    auto best = planner_->bestTeb();
    EXPECT_NE(best, nullptr);
    if (best) {
      EXPECT_TRUE(best->isOptimized());
    }
    return best;
  }

  nav2_util::LifecycleNode::SharedPtr node_;
  teb_local_planner::TebConfig cfg_;
  teb_local_planner::ObstContainer obstacles_;
  std::unique_ptr<teb_local_planner::HomotopyClassPlanner> planner_;
};

TEST_F(PlannedResultExportTest, ParameterizedPlannerStartsInvalidated) {
  teb_local_planner::TebOptimalPlanner planner(node_, cfg_, &obstacles_);

  EXPECT_FALSE(planner.isOptimized());
  std::vector<teb_local_planner::PoseSE2> poses{
      teb_local_planner::PoseSE2(99.0, 99.0, 0.0)};
  std::vector<double> time_diffs{99.0};
  EXPECT_FALSE(planner.getPlannedResult(poses, time_diffs));
  EXPECT_TRUE(poses.empty());
  EXPECT_TRUE(time_diffs.empty());
}

TEST_F(PlannedResultExportTest, HomotopyExportsExactSelectedBand) {
  const auto best = planSuccessfully();
  ASSERT_NE(best, nullptr);

  std::vector<teb_local_planner::PoseSE2> poses;
  std::vector<double> time_diffs;
  ASSERT_TRUE(planner_->getPlannedResult(poses, time_diffs));

  ASSERT_FALSE(poses.empty());
  ASSERT_EQ(poses.size(), time_diffs.size() + 1U);
  ASSERT_EQ(poses.size(), static_cast<std::size_t>(best->teb().sizePoses()));
  ASSERT_EQ(time_diffs.size(), static_cast<std::size_t>(best->teb().sizeTimeDiffs()));

  for (std::size_t i = 0; i < poses.size(); ++i) {
    EXPECT_DOUBLE_EQ(poses[i].x(), best->teb().Pose(i).x());
    EXPECT_DOUBLE_EQ(poses[i].y(), best->teb().Pose(i).y());
    EXPECT_DOUBLE_EQ(poses[i].theta(), best->teb().Pose(i).theta());
  }
  for (std::size_t i = 0; i < time_diffs.size(); ++i) {
    EXPECT_DOUBLE_EQ(time_diffs[i], best->teb().TimeDiff(i));
  }
}

TEST_F(PlannedResultExportTest, FailedOptimizationClearsPreviousExport) {
  const auto old_best = planSuccessfully();
  ASSERT_NE(old_best, nullptr);

  EXPECT_FALSE(old_best->optimizeTEB(cfg_.optim.no_inner_iterations, 0, true));
  EXPECT_FALSE(old_best->isOptimized());

  std::vector<teb_local_planner::PoseSE2> poses{teb_local_planner::PoseSE2(99.0, 99.0, 0.0)};
  std::vector<double> time_diffs{99.0};
  EXPECT_FALSE(planner_->getPlannedResult(poses, time_diffs));
  EXPECT_TRUE(poses.empty());
  EXPECT_TRUE(time_diffs.empty());
}

TEST_F(PlannedResultExportTest, ReinitializeClearsPreviousExport) {
  const auto best = planSuccessfully();
  ASSERT_NE(best, nullptr);

  best->initialize(node_, cfg_, &obstacles_);

  EXPECT_FALSE(best->isOptimized());
  std::vector<teb_local_planner::PoseSE2> poses{
      teb_local_planner::PoseSE2(99.0, 99.0, 0.0)};
  std::vector<double> time_diffs{99.0};
  EXPECT_FALSE(planner_->getPlannedResult(poses, time_diffs));
  EXPECT_TRUE(poses.empty());
  EXPECT_TRUE(time_diffs.empty());
}

TEST_F(PlannedResultExportTest, ClearPlannerClearsPreviousExport) {
  const auto best = planSuccessfully();
  ASSERT_NE(best, nullptr);

  best->clearPlanner();

  std::vector<teb_local_planner::PoseSE2> poses{teb_local_planner::PoseSE2(99.0, 99.0, 0.0)};
  std::vector<double> time_diffs{99.0};
  EXPECT_FALSE(planner_->getPlannedResult(poses, time_diffs));
  EXPECT_TRUE(poses.empty());
  EXPECT_TRUE(time_diffs.empty());
  EXPECT_FALSE(best->isOptimized());
}

}  // namespace
