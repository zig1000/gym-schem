import gymnasium as gym

# Affected by https://github.com/Farama-Foundation/Gymnasium/issues/102
class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action, e.g. for compatability with stable-baselines3."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        # Due to https://github.com/Farama-Foundation/Gymnasium/issues/102, we need to one-hotify the parts of the
        # action that are about to be converted back to a Discrete. Or at least make sure they aren't all 0s.
        if all(action[i] == 0 for i in range(5)):  # Arrow
            action[0] = 1
        # No idea what's getting put in index 5
        if all(action[i] == 0 for i in range(6, 21)):  # Instruction
            action[6] = 1
        if all(action[i] == 0 for i in range(-6, -2)):  # Command-direction
            action[-6] = 1

        try:
            return gym.spaces.utils.unflatten(self.env.action_space, action)
        except ValueError:
            print(action)
            raise
