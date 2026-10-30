"""Pure fixed-list route sampling shared by RL episode reset tests and ROS."""

import math
import random


def sample_list_route(start_poses, goals, minimum_goal_distance,
                      reverse_list_routes, rng=random, *,
                      minimum_route_distance=None):
    """Choose one valid list route and optionally reverse its two endpoints.

    Reversing an already selected pair preserves the authored route set. It is
    intentionally different from mixing both coordinate lists, which would
    introduce start-to-start and goal-to-goal episodes with unknown clearance.
    """
    pose = rng.choice(start_poses)
    start = (float(pose[0]), float(pose[1]))
    route_distance = max(
        minimum_goal_distance,
        minimum_route_distance if minimum_route_distance is not None
        else minimum_goal_distance)
    candidates = [tuple(goal) for goal in goals
                  if math.hypot(goal[0] - start[0], goal[1] - start[1])
                  >= route_distance]
    if not candidates:
        raise RuntimeError(
            f'every goal is closer than {route_distance} m to '
            f'the start pose ({start[0]:.2f}, {start[1]:.2f}). Add a '
            f'farther goal or lower minimum_goal_distance.')
    goal = rng.choice(candidates)
    if reverse_list_routes and rng.random() < 0.5:
        return goal, start
    return start, goal


def route_start_yaw(start, goal, yaw_noise=0.0, rng=random):
    """Aim a route-starting robot at its goal, with bounded yaw noise."""
    if not 0.0 <= yaw_noise <= math.pi / 2.0:
        raise ValueError('talking_start_yaw_noise must be in [0, pi/2] radians')
    heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    return heading + rng.uniform(-yaw_noise, yaw_noise)


def route_seed_frame(goal_frame, waiting_route, list_route):
    """Return the frame used to seed an explicitly authored route.

    A fixed route's start, goal and initial heading form one geometric
    contract.  Seed its localization in `goal_frame`, otherwise an EKF reset
    in `map` can report a different odom pose and make a valid route look as
    though it starts at its goal.  Freely sampled episodes retain the legacy
    map-frame seed because they sample their goal after reset.
    """
    if waiting_route is not None or list_route is not None:
        return goal_frame
    return 'map'
