import numpy as np
import collections
from robel.dkitty.orient import DKittyOrientRandom
from robel.dkitty.walk import BaseDKittyWalk
from robel.simulation.randomize import SimRandomizer
from typing import Dict, Optional, Sequence, Tuple, Union

class DKittyWalkRandom(BaseDKittyWalk):
    """Walk straight towards a random location."""

    def __init__(
            self,
            *args,
            target_distance_range: Tuple[float, float] = (1.0, 2.0),
            # +/- 60deg
            target_angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
            **kwargs):
        """Initializes the environment.

        Args:
            target_distance_range: The range in which to sample the target
                distance.
            target_angle_range: The range in which to sample the angle between
                the initial D'Kitty heading and the target.
        """
        super().__init__(*args, **kwargs)
        self._target_distance_range = target_distance_range
        self._target_angle_range = target_angle_range

    def _reset(self):
        """Resets the environment."""
        target_dist = self.np_random.uniform(*self._target_distance_range)
        # Offset the angle by 90deg since D'Kitty looks towards +y-axis.
        target_theta = np.pi / 2 + np.pi #self.np_random.uniform(
            #*self._target_angle_range)
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])
        super()._reset()

    def get_reward_dict(self, action: np.ndarray, obs_dict: Dict[str, np.ndarray],) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        target_xy_dist = np.linalg.norm(obs_dict['target_error'])
        heading = obs_dict['heading']

        reward_dict = collections.OrderedDict((
            # Add reward terms for being upright.
            *self._get_upright_rewards(obs_dict).items(),
            # Reward for proximity to the target.
            ('target_dist_cost', -4 * target_xy_dist),
            # Heading - 1 @ cos(0) to 0 @ cos(25deg).
            ('heading', 2 * (-heading - 0.9) / 0.1),
            # Bonus
            ('bonus_small', 5 * ((target_xy_dist < 0.5) + (heading > 0.9))),
            ('bonus_big', 10 * (target_xy_dist < 0.5) * (heading > 0.9)),
        ))
        return reward_dict

class DKittyOrientRandomDynamics(DKittyOrientRandom):
    """Walk straight towards a random location."""

    def __init__(self,
                 coef,
                 *args, 
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())
        self.coef = coef

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2*self.coef[0]),
            friction_loss_range=(0.001, 0.005*self.coef[1]),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2*self.coef[2]),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2*self.coef[3]),
            friction_spin_range=(0.003, 0.007*self.coef[4]),
            friction_roll_range=(0.00005, 0.00015*self.coef[5]),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0*self.coef[6]),
            height_field_range=(0, 0.00*self.coef[7]),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()