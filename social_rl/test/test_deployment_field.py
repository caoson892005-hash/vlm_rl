"""Tests for deploy field generation independent of a navigation goal."""

from types import SimpleNamespace
import math

from social_rl.constraint_field import (ConstraintField, ConstraintFieldConfig,
                                        Zone, ZoneSample, compile_zones)
from social_rl.constraint_field_node import (DeploymentConstraintFieldNode,
                                             LockedTalkingField,
                                             waiting_only_field)
from social_rl.observation import RelativeEntity
from social_rl.ros_interface import PerceptionBridge


class _Memory:
    def __init__(self):
        self.received = None
        self.cleared = False

    def remember(self, people, transform, now):
        self.received = (people, transform, now)
        return people

    def clear(self):
        self.cleared = True


class _Cache:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


def test_deploy_field_matches_direct_compile_without_goal():
    people = [
        RelativeEntity(
            x=1.0, y=-0.4, scene_type='talking',
            track_id='person_1', scene_confidence=1.0),
        RelativeEntity(
            x=1.0, y=0.4, scene_type='talking',
            track_id='person_2', scene_confidence=1.0),
    ]
    config = ConstraintFieldConfig()
    memory = _Memory()
    bridge = object.__new__(PerceptionBridge)
    bridge._people_provider = None
    bridge._observation = SimpleNamespace(constraint_field=config)
    bridge._memory = memory
    bridge._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=2_000_000_000)))
    bridge.people_transform_valid = True
    transform = object()
    bridge.goal_transform = lambda: transform
    bridge.relative_people = lambda: people

    field = bridge.deployment_constraint_field()

    assert field == compile_zones(people, config)
    assert memory.received == (people, transform, 2.0)


def test_ground_truth_deploy_field_keeps_waiting_zone():
    """Simulation deploy must feed the same waiting label PPO saw in eval."""
    people = [
        RelativeEntity(
            x=1.0, y=0.0, scene_type='waiting', track_id='person_1',
            scene_confidence=1.0, object_distance=1.4),
    ]
    config = ConstraintFieldConfig()
    memory = _Memory()
    provider = SimpleNamespace(relative_people=lambda apply_camera=True: people)
    bridge = object.__new__(PerceptionBridge)
    bridge._people_provider = provider
    bridge._observation = SimpleNamespace(constraint_field=config)
    bridge._env = SimpleNamespace(people_occlusion=False)
    bridge._memory = memory
    bridge._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=2_000_000_000)))
    bridge.people_transform_valid = False
    transform = object()
    bridge.goal_transform = lambda: transform

    field = bridge.deployment_constraint_field()

    assert field == compile_zones(people, config)
    assert field.zones[0].scene_type == 'waiting'
    assert bridge.people_transform_valid


def _transform(translation_x, yaw=0.0):
    return SimpleNamespace(
        translation=SimpleNamespace(x=translation_x, y=0.0),
        rotation=SimpleNamespace(
            x=0.0, y=0.0,
            z=math.sin(0.5 * yaw), w=math.cos(0.5 * yaw)))


def _field(center, *, scene_type='talking', hardness='hard'):
    sample = ZoneSample(
        t=0.0, shape='gaussian', center=center,
        size=(0.5, 0.2, 0.5), weight=0.9, orientation=0.0)
    zone = Zone(
        zone_id='zone', track_ids=('one', 'two'),
        scene_type=scene_type, hardness=hardness, confidence=1.0,
        valid_from=0.0, valid_to=0.0, trajectory_of_zone=(sample,))
    return ConstraintField(
        timestamp=0.0, frame='base_link', horizon_steps=1, dt=0.0,
        zones=(zone,))


def test_first_talking_zone_is_fixed_in_odom_and_cannot_be_replaced():
    latch = LockedTalkingField('odom', 'base_link')

    # Robot is at odom x=2, so base_link x=1 is odom x=3.
    assert latch.try_lock(_field((1.0, 0.0)), _transform(-2.0), 10.0)
    # A later VLM result must not replace the first accepted constraint.
    assert not latch.try_lock(
        _field((50.0, 0.0)), _transform(-2.0), 12.0)

    fixed = latch.for_visualization()
    assert fixed.frame == 'odom'
    assert fixed.zones[0].trajectory_of_zone[0].center == (3.0, 0.0)

    # After the robot moves to odom x=2.5, the same world zone is 0.5 m ahead.
    output = latch.for_robot(_transform(-2.5))
    assert output.frame == 'base_link'
    assert len(output.zones) == 1
    assert output.zones[0].trajectory_of_zone[0].center == (0.5, 0.0)

    assert not latch.expire_if_due(109.999)
    assert latch.expire_if_due(110.0)
    assert not latch.locked
    assert latch.try_lock(_field((4.0, 0.0)), _transform(0.0), 111.0)


def test_talking_hold_starts_when_first_member_leaves_camera():
    latch = LockedTalkingField('odom', 'base_link')
    observed_at = {'one': 10.0, 'two': 10.0}
    assert latch.try_lock(
        _field((1.0, 0.0)), _transform(0.0), 10.0, observed_at)

    # Both detections renew the clock; a coasted /people track does not.
    observed_at.update(one=50.0, two=50.0)
    latch.note_observations(observed_at, 50.0)
    observed_at['one'] = 80.0
    latch.note_observations(observed_at, 80.0)
    assert not latch.expire_if_due(149.999)
    assert latch.expire_if_due(150.0)


def test_missing_pair_observation_cannot_extend_talking_hold():
    latch = LockedTalkingField('odom', 'base_link')
    assert latch.try_lock(_field((1.0, 0.0)), _transform(0.0), 10.0)
    latch.note_observations({'one': 90.0}, 90.0)
    assert latch.expire_if_due(110.0)


def test_expired_talking_pair_cannot_relock_from_people_memory():
    latch = LockedTalkingField('odom', 'base_link')
    pair = _field((1.0, 0.0))
    observed_at = {'one': 10.0, 'two': 10.0}
    assert latch.try_lock(pair, _transform(0.0), 10.0, observed_at)
    assert latch.expire_if_due(110.0)
    assert not latch.try_lock(pair, _transform(0.0), 110.0, observed_at)
    assert not latch.try_lock(pair, _transform(0.0), 110.0, {'one': 110.0})
    observed_at.update(one=110.0, two=110.0)
    assert latch.try_lock(pair, _transform(0.0), 110.0, observed_at)


def test_non_talking_zone_does_not_lock_the_field():
    latch = LockedTalkingField('odom', 'base_link')

    assert not latch.try_lock(
        _field((1.0, 0.0), scene_type='', hardness='soft'),
        _transform(0.0), 0.0)
    assert not latch.locked


def test_waiting_only_field_keeps_single_person_waiting_zone():
    """Deployment must render a VLM-confirmed waiting person without a pair."""
    waiting = _field((1.0, 0.0), scene_type='waiting', hardness='soft').zones[0]
    passing = _field((2.0, 0.0), scene_type='passing', hardness='soft').zones[0]
    candidate = ConstraintField(
        timestamp=0.0, frame='base_link', horizon_steps=1, dt=0.0,
        zones=(waiting, passing))

    field = waiting_only_field(candidate)

    assert field.zones == (waiting,)


def test_waiting_hold_expires_from_last_camera_observation():
    waiting = _field((1.0, 0.0), scene_type='waiting', hardness='soft').zones[0]
    field = ConstraintField(
        timestamp=0.0, frame='base_link', horizon_steps=1, dt=0.0,
        zones=(waiting,))

    assert waiting_only_field(field, {'one': 10.0}, 109.999).zones == (waiting,)
    assert waiting_only_field(field, {'one': 10.0}, 110.0).zones == ()
    assert waiting_only_field(field, {}, 11.0).zones == ()


def test_camera_observations_do_not_renew_missing_people():
    node = SimpleNamespace(_observed_at={})
    stamp = lambda seconds: SimpleNamespace(sec=seconds, nanosec=0)
    message = lambda seconds, people: SimpleNamespace(
        header=SimpleNamespace(stamp=stamp(seconds)),
        people=[SimpleNamespace(id=person_id) for person_id in people])

    DeploymentConstraintFieldNode._on_observations(
        node, message(10, ['one', 'two']))
    DeploymentConstraintFieldNode._on_observations(node, message(11, ['one']))

    assert node._observed_at == {'one': 11.0, 'two': 10.0}


def test_deployment_reset_clears_people_memory_and_vlm_cache():
    bridge = object.__new__(PerceptionBridge)
    bridge._people_provider = None
    bridge._memory = _Memory()
    bridge._vlm_states = _Cache()
    bridge.constraint_field = object()

    bridge.reset_deployment_grounding()

    assert bridge._memory.cleared
    assert bridge._vlm_states.cleared
    assert bridge.constraint_field is None
