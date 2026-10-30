"""Regression checks for fixed-list route reversal without a ROS graph."""

import unittest

from social_rl.route_sampling import (route_seed_frame, route_start_yaw,
                                      sample_list_route)


class FixedRandom:
    """Minimal deterministic random source for route-direction checks."""

    def __init__(self, random_value):
        self.random_value = random_value

    @staticmethod
    def choice(values):
        return values[0]

    def random(self):
        return self.random_value


class RouteReversalTest(unittest.TestCase):
    def setUp(self):
        self.start_poses = [[0.0, 0.0, 0.0]]
        self.goals = [[3.0, 0.0]]

    def test_reverse_draw_swaps_only_the_selected_route_ends(self):
        start, goal = sample_list_route(
            self.start_poses, self.goals, 2.0, True, FixedRandom(0.25))

        self.assertEqual(start, (3.0, 0.0))
        self.assertEqual(goal, (0.0, 0.0))

    def test_forward_draw_keeps_the_selected_route_direction(self):
        start, goal = sample_list_route(
            self.start_poses, self.goals, 2.0, True, FixedRandom(0.75))

        self.assertEqual(start, (0.0, 0.0))
        self.assertEqual(goal, (3.0, 0.0))

    def test_reversed_route_starts_facing_its_goal(self):
        yaw = route_start_yaw((3.0, 0.0), (0.0, 0.0))

        self.assertAlmostEqual(yaw, 3.141592653589793)

    def test_talking_minimum_route_distance_filters_short_routes(self):
        start, goal = sample_list_route(
            self.start_poses, [[2.5, 0.0], [4.0, 0.0]], 2.0, False,
            FixedRandom(0.75), minimum_route_distance=3.5)

        self.assertEqual(start, (0.0, 0.0))
        self.assertEqual(goal, (4.0, 0.0))

    def test_fixed_list_route_seeds_in_its_goal_frame(self):
        frame = route_seed_frame('odom', None, ((3.0, 0.0), (0.0, 0.0)))

        self.assertEqual(frame, 'odom')

    def test_free_space_route_keeps_legacy_map_seed(self):
        frame = route_seed_frame('odom', None, None)

        self.assertEqual(frame, 'map')


if __name__ == '__main__':
    unittest.main()
