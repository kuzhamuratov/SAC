from robel.dkitty.orient import DKittyOrientRandom
from robel.simulation.randomize import SimRandomizer
from typing import Optional


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
            height_field_range=(0, 0.05*self.coef[7]),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()