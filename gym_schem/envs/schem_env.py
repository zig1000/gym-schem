from enum import IntEnum

import gymnasium as gym
from gymnasium import error, spaces
import schem
#import numpy as np


class SolutionPlacementPhase(IntEnum):
    FEATURE = 0  # posn output per feature
    RED_START_POSN = 1
    RED_START_DIRN = 2
    BLUE_START_POSN = 3
    BLU_START_DIRN = 4
    RED_ARROW = 5
    BLUE_ARROW = 6
    RED_INSTANT = 7
    BLUE_INSTANT = 8


class SChemEnvJustInTime(gym.Env):
    """An SChemEnv implementation in which solution symbols are placed just-in-time while the solution is being run.
    When revisiting a grid cell in which the agent was already given the opportunity to place a symbol (whether or not
    they did), that cycle is fast-forwarded past, with step() potentially simulating multiple cycles per step.

    In full, a run proceeds as follows:
    first N step() calls: agent must select a (col, row) coordinate to place each feature in turn until all N features
                          (bonders, sensor, fuser, etc.) of the level are placed.
    Next 4 step() calls: agent must select a coordinate for the red waldo's start command, followed by a cardinal
                         direction for its starting orientation, then repeat these choices for the blue waldo.
    Remaining step() calls: Each cycle, agent must make up to 4 step() action choices, in the order:
        red arrow, blue arrow, red instant action, blue instant action. The observation space will include a value
        indicating which phase the current step() is in, and will skip a waldo's step()'s during a cycle if it is in a
        grid cell it has already visited (e.g. flip-flops can't be placed only on the second time a grid cell is
        visited, even though they wouldn't have affected the first pass).
        Also note that the Env.action_space is also a dynamic property updated accordingly, such that sampling from it
        will always give a valid action.
    """
    metadata = {'render.modes': ['human']}
    last_level_solved = -1
    max_training_level

    def __init__(self):
        self.observation_space = spaces.Box([2, 2])
        self.level = schem.Level(schem.levels["Of Pancakes and Spaceships"])
        self.solution = schem.Solution(level=self.Level)
        self._visited_posns_red = set()
        self._visited_posns_blue = set()

        # a default empty solution

        self.observation_space = spaces.Box([2, 2])
        self.phase = 'feature-placement'

    @property
    def action_space(self):
        if self.phase = 0:
            pass
        # Calculate available actions dynamically depending what point we're at
        return spaces.Discrete(14)

    def step(self, action):
        # TODO: Feature placement phase

        state, reward, done, info = [], 0, False, {}
        return state, reward, done, info

    def reset(self):
        # TODO: Change levels if the agent beat the current level?
        self.solution.reset()
        state = 0
        return state

    def render(self):
        """Pretty-print the current reactor state human-readably."""
        pass


class SChemEnvStepWise(gym.Env):
    """An SChem env where the input is the current partially-contructed solution, and the agent must place or remove
    another symbol or feature. The solution will be run after every change, returning a reward of 0 if the solution
    does not complete or has not improved its best score, and otherwise returning the difference in how much the
    solution was improved.
    """
    pass


class SChemEnvOneShot(gym.Env):
    """An SChem env where the input is the level json, and the output is the solution export string.
    There is only a single step, with the observations being the per-cycle frames of the solution's run.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Box([8, 10, ])
        self.observation_space = spaces.Box([2, 2])
        self.level = schem.Level(schem.levels["Of Pancakes and Spaceships"])
        self.solution = schem.Solution(level=self.Level)

    def solution_to_observation(self, solution, include_static_layers=True):
        """Given a research schem.Solution object currently being run, return a numpy array representing its current
        state. TODO: Support production levels too
        Layers:
        - 8x10 representing atoms, values are 0-109 normalized to 0-1.
        - 8x10 representing rightward bonds (last col is always 0), values are 0-3 normalized to 0-1.
        - 8x10 representing downward bonds (last row is always 0), values are 0-3 normalized to 0-1.
        - two normalized 0-1 values representing the number of outputs currently completed for each output zone.
        - two 8x10's representing respective waldo positions (1 if present, 0 if not).
        - 4 bits representing waldo directions
        -
        - [Optional] Static layers representing solution entities that were already placed by the agent or included in
                     the puzzle definition.
          These might improve an agent's ability to associate its actions to their effects on the environment,
          but they also greatly increase the size of the observation space due to the necessity of one-hot encoding.
          Set include_static_layers=False to disable.
          - three 8x10 input/output atom layers, encoded in the same way as the active molecules layer, with
            inputs/outputs in their respective zones and the middle columns unused.
          - two 8x10 input/output right/down bond layers, similarly to the active molecules layer.
          - Six 8x10 layers representing the locations of features (bonders+, bonders-, sensors, fusers, splitters,
            tunnels). Regular +/- bonders have a 1 in both + and - layers. Double-wide features have only the left
            position encoded (TODO: should they have both?).
            Priority orders are consistent with the order of placement but not represented in observations.
          - Thirty-two (2x16) 8x10 layers representing one-hot encodings of the currently placed symbols for each waldo.
            The sense layer will use normalized 0-1 values representing the sensor element, similarly to the
            representation of atoms.
        """
        pass

    @property
    def action_space(self):
        if self.phase = 0:
            pass
        # Calculate available actions dynamically depending what point we're at
        return spaces.Discrete(14)

    def step(self, action):
        """Given a solution string, load and run the solution. This is the environment's only step.

        Returns (state, reward, done, info), of which done is always True.
        """

        state, reward, done, info = [], 0, True, {}
        return state, reward, True, info

    def reset(self):
        # TODO: Change levels if the agent beat the current level?
        self.solution.reset()
        state = 0
        return state

    def render(self):
        """Pretty-print the current reactor state human-readably."""
        pass
