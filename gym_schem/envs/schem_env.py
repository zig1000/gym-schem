import copy
import math
from typing import Optional

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Box, Dict, Discrete, Tuple
import numpy as np
import rich
import schem
from schem import *
from schem.components import Reactor, RandomInput
from schem.grid import Direction, Position, CARDINAL_DIRECTIONS, RIGHT, DOWN
from schem.waldo import Waldo, Instruction, InstructionType

from gym_schem.utils.human_scores import top_human_score_weighted

ROTATIONAL_DIRECTIONS = (Direction.CLOCKWISE, Direction.COUNTER_CLOCKWISE)
NUM_WALDOS, NUM_COLS, NUM_ROWS = Reactor.NUM_WALDOS, Reactor.NUM_COLS, Reactor.NUM_ROWS
INSTR_IDX_TO_TYPE = list(InstructionType)  # Mapping of instruction action indices to schem's native enum
INSTR_TYPE_TO_IDX = {t: i for i, t in enumerate(InstructionType)}
# Note that 'bonder' is not included as we're going to demark it by both bond+ and bond- being selected.
FEATURE_IDX_TO_NAME = ['bonder-plus', 'bonder-minus', 'sensor', 'fuser', 'splitter', 'tunnel']

# Parts of the observation space that are common across env types
# TODO: Scale to handle production levels
# Current space size (ignoring types): 3 + 480 + 4 + 128 + 128 + 320 + 168 + 640 + 2400 + 640 + 160 + 160 = 5231
shared_observation_dict = {
    'optimization_target': Box(shape=(3,), dtype=float, low=0, high=1),  # Cycles, Reactors, Symbols
    # One-hot encoding of all reactor features, except +- bonders which are stored as a 1 in both the + and -
    # bonder sections: (bonder+, bonder-, sensor, fuser, splitter, swapper).
    # Could consider scaling the value based on feature priority (i.e. show the agent bonder priorities), but
    # I'm kind of fine with forcing agents to work that out themselves in order to keep the space simpler.
    'features': Box(shape=(NUM_ROWS, NUM_COLS, 6), dtype=bool, low=0, high=1),
    'target_output_counts': Box(shape=(2,), dtype=np.int8, low=0, high=127),
    'current_output_counts': Box(shape=(2,), dtype=np.int8, low=0, high=127),
    # Input molecule details (both zones)
    'input_atoms': Box(shape=(2, 4, 4), dtype=np.int16, low=0, high=204),
    'input_atoms_max_bonds': Box(shape=(2, 4, 4), dtype=np.int8, low=0, high=12),
    'input_bonds': Box(shape=(2, 4, 4, 2), dtype=np.int8, low=0, high=3),  # (right, down)
    # Output molecule details (both zones)
    'output_atoms': Box(shape=(2, 4, 4), dtype=np.int16, low=0, high=204),
    'output_atoms_max_bonds': Box(shape=(2, 4, 4), dtype=np.int8, low=0, high=12),
    'output_bonds': Box(shape=(2, 4, 4, 2), dtype=np.int8, low=0, high=3),  # (right, down)
    # Molecules currently in the reactor
    'atoms': Box(shape=(NUM_ROWS, NUM_COLS), dtype=np.int16, low=0, high=204),  # 0 = no atom
    'atoms_max_bonds': Box(shape=(NUM_ROWS, NUM_COLS), dtype=np.int8, low=0, high=12),
    'bonds': Box(shape=(NUM_ROWS, NUM_COLS, 2), dtype=np.int8, low=0, high=3),  # (right, down)
    # Mid-run waldo attributes. One-hot encoded; observation_space.sample() will behave poorly...
    # TODO: MultiDiscrete makes sample() behave well (can't have two directions at once unlike if we made a
    #       naive one-hot-encoding-shaped Box and used sample() on it), but the actual returned observation is
    #       not one-hot encoded (e.g. confusing discussion: https://github.com/openai/gym/issues/3157).
    #       I think I prefer pre-one-hot encoding the observation but research/ask this.
    'waldo_positions': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS), dtype=bool, low=0, high=1),
    'waldo_directions': Box(shape=(NUM_WALDOS, 4), dtype=bool, low=0, high=1),
    # Waldo instructions - these are static and large and can be optionally excluded
    # Arrows: One-hot encoded clockwise from UP = 0. Note that the one-hot encoding can have 0 for all values
    #         to represent no arrow in the cell.
    'waldo_arrows': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS, 4), dtype=bool, low=0, high=1),
    # Commands: one-hot encoded, 0 for all = no command. -2 to exclude Pause and Control
    'waldo_commands': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS, len(schem.waldo.InstructionType) - 2),
                          dtype=bool, low=0, high=1),
    # Direction associated with a command (NOT the cell's arrow). Only applicable for Start, Sense, Flip-Flop,
    # and Rotate. For Rotate, the UP & DOWN (0th & 2nd) encodings represent CLOCKWISE & COUNTER_CLOCKWISE.
    # Actually fuck it I'm going to borrow this for Input alpha vs beta and Output psi vs omega too, again using
    # the UP & DOWN encodings for each to preserve them being 'opposites'.
    'waldo_command_directions': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS, 4), dtype=bool, low=0, high=1),
    # Element associated with Sense commands (note that it can't be Australium)
    'waldo_command_elements': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS), dtype=np.int16, low=0, high=203),
    # 1 if there is an active flip-flop in the given position
    'waldo_flip_flop_states': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS), dtype=bool, low=0, high=1)}


# Populate the parts of the envs' obervation spaces that are shared in shared_observation_dict above.
_instr_type_to_index = {instr_type: i for i, instr_type in enumerate(schem.waldo.InstructionType)}
def _shared_observation(solution: Solution,
                        optimization_goal: tuple[float, float, float],
                        include_waldo_layers=True):
    """Given a research schem.Solution object currently being run, return an observation formatted as defined by
    observation_space. TODO: Support production levels too
    Waldo static layers:
      These might improve an agent's ability to associate its actions to their effects on the environment,
      but they also greatly increase the size of the observation space due to the necessity of one-hot encoding.
      Set include_waldo_layers=False to disable.
      - three 8x10 input/output atom layers, encoded in the same way as the active molecules layer, with
        inputs/outputs in their respective zones and the middle columns unused.
      - two 8x10 input/output right/down bond layers, similarly to the active molecules layer.
      - Six 8x10 layers representing the locations of features (bonders+, bonders-, sensors, fusers, splitters,
        tunnels). Regular +/- bonders have a 1 in both + and - layers. Double-wide features have only the left
        position encoded.
        Priority orders are consistent with the order of placement but not represented in observations.
      - Thirty-two (2x16) 8x10 layers representing one-hot encodings of the currently placed symbols for each waldo.
        The sense layer will use normalized 0-1 values representing the sensor element, similarly to the
        representation of atoms.
    """
    #flattened_space = gym.spaces.utils.flatten_space(self.observation_space)
    obs = {space_key: np.zeros(shape=space.shape, dtype=space.dtype)
           for space_key, space in shared_observation_dict.items()}
    reactor = next(solution.reactors)

    for i, metric_weight in enumerate(optimization_goal):
        obs['optimization_target'][i] = metric_weight

    # schem handles bonders differently since they mix priorities with plus vs minus bonders; store the correct type
    # (and store as both + and - for regular +- bonders).
    # For now I'm hiding feature priorities from the agent like OG SC.
    for (c, r), bond_type in reactor.bonders:
        if '+' in bond_type:
            obs['features'][r][c][0] = 1
        if '-' in bond_type:
            obs['features'][r][c][1] = 1

    for i, feature in enumerate(('sensors', 'fusers', 'splitters', 'swappers')):
        for c, r in getattr(reactor, feature):
            # Note that fusers and splitters store only their left position. This is probably preferable.
            obs['features'][r][c][i + 3] = 1

    for i, input in enumerate(solution.inputs):
        for (c, r), atom in input.molecules[0].atom_map.items():
            obs['input_atoms'][i][r][c] = atom.element.atomic_num
            obs['input_atoms_max_bonds'][i][r][c] = atom.element.max_bonds
            obs['input_bonds'][i][r][c][0] = atom.bonds[RIGHT] if RIGHT in atom.bonds else 0
            obs['input_bonds'][i][r][c][1] = atom.bonds[DOWN] if DOWN in atom.bonds else 0

    for i, output in enumerate(solution.outputs):
        obs['target_output_counts'][i] = output.target_count
        obs['current_output_counts'][i] = output.current_count
        for (c, r), atom in output.output_molecule.atom_map.items():
            obs['output_atoms'][i][r][c] = atom.element.atomic_num
            obs['output_atoms_max_bonds'][i][r][c] = atom.element.max_bonds
            obs['output_bonds'][i]
            obs['output_bonds'][i][r][c][0] = atom.bonds[RIGHT] if RIGHT in atom.bonds else 0
            obs['output_bonds'][i][r][c][1] = atom.bonds[DOWN] if DOWN in atom.bonds else 0

    for molecule in reactor.molecules:
        for (c, r), atom in molecule.atom_map.items():
            obs['atoms'][r][c] = atom.element.atomic_num
            obs['atoms_max_bonds'][r][c] = atom.element.max_bonds
            obs['bonds'][r][c][0] = atom.bonds[RIGHT] if RIGHT in atom.bonds else 0
            obs['bonds'][r][c][1] = atom.bonds[DOWN] if DOWN in atom.bonds else 0

    if include_waldo_layers:
        for w, waldo in enumerate(reactor.waldos):
            obs['waldo_positions'][w][waldo.position.row][waldo.position.col] = 1
            obs['waldo_directions'][w][waldo.direction.value] = 1
            for (c, r), arrow in waldo.arrows.items():
                obs['waldo_arrows'][w][r][c][arrow.value] = 1
            for (c, r), cmd in waldo.commands.items():
                obs['waldo_commands'][w][r][c][_instr_type_to_index[cmd.type]] = 1
                if cmd.direction is not None:
                    # Convert rotate's CLOCKWISE = 5 and COUNTERCLOCKWISE = 7 to 0 and 2 respectively
                    dirn_idx = cmd.direction.value if cmd.direction.value < 4 else cmd.direction.value - 5
                    obs['waldo_command_directions'][w][r][c][dirn_idx] = 1

                if cmd.type == schem.waldo.InstructionType.SENSE:
                    obs['waldo_command_elements'][w][r][c] = cmd.target_idx
                elif cmd.target_idx is not None:  # Input/Ouput: 0 (UP) for alpha/psi, 2 (DOWN) for beta/omega
                    obs['waldo_command_directions'][w][r][c][2 * cmd.target_idx] = 1

            for (c, r), active in waldo.flipflop_states.items():
                if active:
                    obs['waldo_flip_flop_states'][w][r][c] = 1

    return obs


class SChemEnv(gym.Env):
    """An SChem env implementation in which solution symbols are placed just-in-time while the solution is being run.

    In full, a run proceeds as follows:
    * First N steps: Agent must select a (col, row) coordinate to place each feature in turn until all N 'features'
      (bonders, sensor, fuser, etc.) of the level are placed (the first level has 0 features).
    * Next 2 steps: Agent must select a coordinate for the current waldo's start location, along with the Start
      command's direction and any arrow for the starting cell.
    * Remaining steps: Each cycle of the solution has 2 steps: the agent alternates placement for each waldo
        (red then blue), placing up to one command and one arrow in the waldo's current cell to be immediately
        executed. Reactor movement is also performed at the end of blue steps.
        The observation space includes a value indicating which waldo the current step is for, and ignores the action if
        it is in a grid cell the current waldo has previously passed through.
        TODO: In future placing flip-flops may be allowed on the second pass through a cell.
    """
    metadata = {'render_modes': ['human']}
    reward_range = (-10, 2)  # Max is 1, realistically
    observation_space = Dict({**shared_observation_dict,
                              # Number of each feature the solution will include
                              'total_features': Box(shape=(6,), dtype=np.int8, low=0, high=8),
                              # Helper feature so the agent doesn't have to learn what order features get
                              # placed in. Indicates which feature is about to be placed.
                              # Same encoding as placed features:
                              # (none, bonder+, bonder-, sensor, fuser, splitter, swapper)
                              # where +- bonders get a 1 in both the + and - slots.
                              'feature_being_placed': Box(shape=(7,), dtype=np.int8, low=0, high=1),
                              # Which waldo is being placed. 0=red, 1=blue. 0 during feature placement.
                              'waldo_being_placed': Discrete(2)})
    action_space = Dict({
        # col/row only used during feature and waldo Start placement
        'col': Box(shape=(1,), dtype=np.int8, low=0, high=Reactor.NUM_COLS - 1),
        'row': Box(shape=(1,), dtype=np.int8, low=0, high=Reactor.NUM_ROWS - 1),
        'arrow': Discrete(5),  # NONE, UP, RIGHT, DOWN, LEFT
        'command': Discrete(len(InstructionType) - 2),  # 0 = no command, and we exclude START, CTRL, and PAUSE
        'command_direction': Discrete(4),  # Used by Start, Sense, Flip-Flop, Input, and Output
        # Box not Discrete since atomic number is in some sense 'ordered'?
        'command_element': Box(shape=(1,), dtype=np.int16, low=1, high=203)})  # Only used by sense.

    # TODO: Accept level or level index, and ensure non-research levels are filtered out
    def __init__(self, optimization_goal=(0.99, 0, 0.01), render_mode=None):
        self.render_mode = render_mode

        # Internal vars
        self.level = schem.Level(next(iter(schem.levels.values())))  # Of Pancakes and Spaceships
        self.solution = schem.Solution(None, level=self.level)
        reactor = next(self.solution.reactors)
        self.optimization_goal = optimization_goal
        self.top_human_score, self.top_human_score_weighted = top_human_score_weighted(self.level.name,
                                                                                       metric_weights=optimization_goal)

        # Check which waldo command action indices are illegal (will be ignored)
        self.disallowed_command_actions = set()
        instr_to_indices = {'instr-bond': {8, 9}, 'instr-sensor': {10}, 'instr-toggle': {11},
                            'instr-fuse': {12}, 'instr-split': {13}, 'instr-swap': {14},
                            'instr-control': set()}  # We exclude PAUSE and CTRL from the action space already
        for disallowed_instr in reactor.disallowed_instrs:
            self.disallowed_command_actions.update(instr_to_indices[disallowed_instr])

        # Need to init some schem-internal loop detection vars, since we're somewhat re-implementing its run() function
        self.solution._random_inputs = [c for c in self.solution.components if isinstance(c, RandomInput)]
        self.solution._random_input_copies = [copy.deepcopy(i) for i in self.solution._random_inputs]

        # Remove solution waldos so initial observation shows Starts missing
        for reactor in self.solution.reactors:
            reactor.waldos = []

        self._visited_posns = [set(), set()]  # Visited posns, indexed by waldo
        self.active_waldo = 0
        self._placed_features = True  # TODO
        self._placed_waldo_starts = False

        # Problem: schem doesn't allow initializing a solution with no features or waldo start commands.
        # We have to 'hide' them from the agent until we're done feature/Start placement.
        # Rip out the reactor's features and Start instructions. We'll put them back, promise.
        expected_features = []
        for reactor in self.solution.reactors:
            expected_features.append(schem.components.REACTOR_TYPES[reactor.type])

            reactor.bonders = []
            reactor.sensors = []
            reactor.fusers = []
            reactor.splitters = []
            reactor.swappers = []

    def _observation(self):
        """Return an observation of the env's current state."""
        obs = _shared_observation(solution=self.solution, optimization_goal=self.optimization_goal,
                                  include_waldo_layers=True)
        # Add parts of the observation specific to this variant of the env
        obs['total_features'] = np.zeros((6,), dtype=np.int8)
        obs['feature_being_placed'] = np.zeros((7,), dtype=np.int8)
        obs['waldo_being_placed'] = self.active_waldo
        return obs

    def fail_reward(self):
        """The reward returned for a failed solution. Override this method to use a custom negative function (or just
        ignore the reward output...).

        By default provides a reward as follows:
        * -10 baseline
        * -10 if no bonders are connected in a level containing bonders.
        * Up to +5 based on number of output zones with at least one successfully outputted molecule.
        * Up to +3 for the closest chemical distance of any board molecule(s) from output zones with 0 successful
                   outputs. The chemical distance will ~= the distance in terms of SC instructions (bond, fuse, etc.),
                   and normalizing for the original chemical distances of the inputs from the outputs.
        * Up to +1 if there is a molecule exactly matching any target output, based on its orthogonal distance from the
                   output zone and whether it has been dropped in the output zone.
        """
        reward = -10
        reactor = next(self.solution.reactors)
        outputs = list(self.solution.outputs)

        if len(reactor.bonders) >= 2 and len(reactor.bond_plus_pairs) == len(reactor.bond_minus_pairs) == 0:
            reward -= 10

        for o, output in enumerate(outputs):
            # Bonus for doing at least one successful output
            if output.current_count > 0:
                reward += 5 / len(outputs)
                continue

            # TODO: Proper chemical distance AKA the hard one. Doesn't affect first level so being naive for now.
            max_molecule_reward = 0
            for molecule in reactor.molecules:
                molecule_reward = 0

                # If the solution crashed while moving/rotating, the molecule will be on float coordinates and our
                # isomoprhism algorithm will explode. The int coordinates are theoretically recoverable, but for now
                # buster out. Note that all posns should be on float coords if any are, so only need to check one.
                if isinstance(next(iter(molecule.atom_map)).col, float):
                    continue

                if molecule.isomorphic(output.output_molecule):
                    molecule_reward += 3  # Bonus for exact output molecule being in grid

                    # Measure distance of furthest atom of molecule from the relevant output zone
                    grid_distance = max(max(6 - posn.col, 0) + (max(posn.row - 3, 0) if o == 0 else max(4 - posn.row))
                                        for posn in molecule.atom_map)
                    molecule_reward += 1 - grid_distance / 10  # 10 is the maximum possible grid distance
                    grabbed = any(waldo.molecule is molecule for waldo in reactor.waldos)
                    # Slightly reward grabbing a molecule outside the output zone and dropping it in it
                    molecule_reward += (0.05 * grabbed) if grid_distance > 0 else (-0.05 * grabbed)

                max_molecule_reward = max(max_molecule_reward, molecule_reward)
            reward += max_molecule_reward

        return reward

    def step(self, action):
        state, reward, terminated, truncated, info = [], 0, False, False, {}
        reactor = next(self.solution.reactors)

        if not self._placed_features:
            # Check how many features we have left to place

            self._placed_features = True
            # Re-init Reactor's bonder pair helper properties
            reactor.bond_plus_pairs, reactor.bond_minus_pairs = reactor.bond_pairs()

            return state, reward, terminated, truncated, info

        # Command/arrow placement (skipped if the current waldo has already visited this cell)
        if (len(reactor.waldos) < 2
                or reactor.waldos[self.active_waldo].position not in self._visited_posns[self.active_waldo]):
            posn = Position(col=action['col'][0], row=action['row'][0])
            dirn = CARDINAL_DIRECTIONS[action['command_direction']]

            # On first placement, initialize this waldo (replacing command with a Start). Otherwise, place a command
            if len(reactor.waldos) < reactor.NUM_WALDOS:
                reactor.waldos.append(Waldo(idx=len(reactor.waldos), arrows={},
                                            commands={posn: Instruction(InstructionType.START, direction=dirn)}))
            elif action['command'] > 0 and action['command'] not in self.disallowed_command_actions:
                instr_type = INSTR_IDX_TO_TYPE[action['command']]  # No-op action replaced Start, so no offset needed
                dirn = (CARDINAL_DIRECTIONS[action['command_direction']]
                        if instr_type != InstructionType.ROTATE
                        else ROTATIONAL_DIRECTIONS[action['command_direction'] % 2])  # up/down = c-wise, right/left = cc-wise

                # Used by input/output and sense.
                target_idx = (action['command_element'][0]
                              if instr_type == InstructionType.SENSE
                              else int(action['command_direction'] > 1))  # Rigged so UP = Alpha and DOWN = Beta

                # TODO: schem should ignore dirn/target on instructions that don't use them; currently it sets them.
                waldo = reactor.waldos[self.active_waldo]
                waldo.commands[waldo.position] = Instruction(instr_type, direction=dirn, target_idx=target_idx)

            waldo = reactor.waldos[self.active_waldo]

            # Place arrow
            waldo = reactor.waldos[self.active_waldo]
            if action['arrow'] > 0:
                waldo.arrows[waldo.position] = CARDINAL_DIRECTIONS[action['arrow'] - 1]

            # Mark the current position as visited by this waldo so we don't overwrite previous steps
            self._visited_posns[self.active_waldo].add(waldo.position)

        # After blue waldo placements, execute movement phase
        if self.active_waldo == 1:
            self.solution.cycle += 1

            try:
                # Instant actions + check for completion. Bit of DRY violation with schem here.
                for component in self.solution.components:
                    if (component.do_instant_actions(self.solution.cycle)  # True if it's an Output that just completed
                            and all(output.current_count >= output.target_count for output in self.solution.outputs)):
                        score = Score(self.solution.cycle - 1, len(list(self.solution.reactors)), self.solution.symbols)
                        # Weighted sum as compared to human best (inverse since weighted score is being minimized)
                        reward = self.top_human_score_weighted / sum(w * m for w, m in zip(self.optimization_goal, score))
                        return state, reward, True, truncated, info

                self.solution.cycle_movement()

                if self.solution.does_it_halt():  # Detects infinite loops and fast-forwards cycles to completion
                    score = Score(self.solution.cycle - 1, len(list(self.solution.reactors)), self.solution.symbols)
                    # Weighted sum as compared to human best (inverse since weighted score is being minimized)
                    reward = self.top_human_score_weighted / sum(w * m for w, m in zip(self.optimization_goal, score))
                    return state, reward, True, truncated, info
            except SolutionRunError as e:  # Any other error would be an error in the env itself
                # If there was a reaction error or an infinite loop occurred, return a negative reward based on the
                # final reactor state.
                info['error'] = str(e)
                # If the solution crashed or an infinite loop occurred, return a negative reward based on the
                # final reactor state.
                return state, self.fail_reward(), True, truncated, info

        self.active_waldo = 1 - self.active_waldo  # Flip waldo

        state = self._observation()
        return state, reward, False, truncated, info

    def reset(self, **kwargs):  # Accept and ignore whatever random-ass shit stable baselines etc. still expect to exist
        """Reset the environment to the beginning of a new episode."""
        # TODO: Change levels if the agent beat the current level?
        self.solution.reset()

        # TODO: clear solution features

        # Remove solution waldos
        for reactor in self.solution.reactors:
            reactor.waldos = []

        self._visited_posns = [set(), set()]
        self.active_waldo = 0
        self._placed_features = True  # TODO
        self._placed_waldo_starts = False

        # Hashing var that's dynamic and thus schem won't reset outside of Solution.run() (which we don't call)
        self.solution._random_input_copies = [copy.deepcopy(i) for i in self.solution._random_inputs]

        return self._observation(), {}

    def render(self, **kwargs):
        """Pretty-print the current reactor state human-readably."""
        rich.print(next(self.solution.reactors).__str__(show_instructions=True, flash_features=False))

    @classmethod
    def solution_to_actions(cls, soln_str: str, level_str: Optional[str] = None):
        """Given a solution export string, return a list of actions that would reproduce it in this env."""
        solution = Solution(soln_str, level=level_str)
        reactor = next(solution.reactors)  # TODO: Productionize
        actions = []

        # TODO: record feature steps

        # Prep schem-internal vars needed for does_it_halt
        solution._random_inputs = [c for c in solution.components if isinstance(c, RandomInput)]
        solution._random_input_copies = [copy.deepcopy(i) for i in solution._random_inputs]

        # Step through the solution cycle by cycle, recording waldo instructions for newly-reached cells
        waldo_visited_cells = [set(), set()]
        while True:
            # Record waldo instructions and arrows
            for w, waldo in enumerate(reactor.waldos):
                action = {space_key: np.zeros(shape=space.shape, dtype=space.dtype)
                          for space_key, space in cls.action_space.items()}
                if not waldo.position in waldo_visited_cells[w]:
                    waldo_visited_cells[w].add(waldo.position)
                    action['col'][0], action['row'][0] = waldo.position  # Only relevant for Start steps but why not
                    if waldo.position in waldo.commands:
                        cmd = waldo.commands[waldo.position]
                        action['command'] = INSTR_TYPE_TO_IDX[cmd.type]
                        if cmd.direction is not None:
                            if cmd.type == InstructionType.ROTATE:
                                action['command_direction'] = int(cmd.direction == Direction.COUNTER_CLOCKWISE)
                            else:
                                action['command_direction'] = cmd.direction.value
                        if cmd.target_idx is not None:
                            # Edge case because we wanted direction to control input/output, which doesn't match schem.
                            if cmd.type in {InstructionType.INPUT, InstructionType.OUTPUT}:
                                action['command_direction'] = cmd.target_idx * 2  # UP for alpha, DOWN for beta
                            else:
                                action['command_element'][0] = cmd.target_idx

                    if waldo.position in waldo.arrows:
                        action['arrow'] = waldo.arrows[waldo.position].value + 1  # Recall that 0 = no arrow

                actions.append(action)

            # Execute cycle
            solution.cycle += 1
            for component in solution.components:
                if (component.do_instant_actions(solution.cycle)  # True if it's an Output that just completed
                        and all(output.current_count >= output.target_count for output in solution.outputs)):
                    return actions

            solution.cycle_movement()

            if solution.does_it_halt():  # Detects infinite loops and fast-forwards cycles to completion
                return actions


# TODO: Allow an already-partially-constructed solution to be used as the input for the OneShot env, even though the
#       solution will always overwrite it with the NN's output? Could serve as a kind of 'guide' for the neural net on
#       what it is 'currently trying'? Or would it just learn to ignore those inputs as they aren't correlated to the
#       reward?
class SChemEnvOneShot(gym.Env):
    """An SChem env with only a single step: the input is an (empty?) solution, the action space is the placement of
    features (bonders etc.) plus all symbols.
    There is only a single step, with the observations being the per-cycle frames of the solution's run.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, optimization_goal=(0.9999, 0, 0.0001)):
        # gym API properties
        self.observation_space = Dict({**shared_observation_dict})  # This env has no special state values

        # TODO: Space.sample() has a bug where dtype=bool always samples as all true, using int8 for now.
        self.action_space = Dict({
            # One-hot encoding of all feature types, except +- bonders are stored as a 1 in both the + and - bonder
            # sections (bonder+, bonder-, sensor, fuser, splitter, swapper)
            # TODO: Scale the value stored in each feature slot based on feature priority. For now, no control of priority.
            'features': Box(shape=(NUM_ROWS, NUM_COLS, 7), dtype=float, low=0, high=1),
            # Arrows: One-hot encoded, 0 = no arrow, after that clockwise from UP.
            'waldo_arrows': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS, 5), dtype=float, low=0, high=1),
            # Commands: One-hot encoded, 0 = no command, and we exclude CTRL and PAUSE
            'waldo_commands': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS, len(InstructionType) - 1),
                                  dtype=float, low=0, high=1),
            # Direction associated with a command (NOT the cell's arrow). Only applicable for Start, Sense, Flip-Flop,
            # and Rotate. For Rotate, the UP & DOWN (0th & 2nd) encodings represent CLOCKWISE & COUNTER_CLOCKWISE.
            # Actually fuck it I'm going to borrow this for Input alpha vs beta and Output psi vs omega too, again using
            # the UP & DOWN encodings for each to preserve them being 'opposites'.
            'waldo_command_directions': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS, 4), dtype=float, low=0, high=1),
            # Element associated with Sense commands (note that sensors can't be set to Australium=204)
            'waldo_command_elements': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS), dtype=np.int16, low=1, high=203)})

        self.level = schem.Level(schem.levels["Of Pancakes and Spaceships"])
        self.solution = schem.Solution(None, level=self.level)
        # Normalize goal so it sums to 1 to prevent reward bloat.
        self.optimization_goal = tuple(subgoal / sum(optimization_goal) for subgoal in optimization_goal)
        self.top_human_score, self.top_human_score_weighted = top_human_score_weighted(self.level.name,
                                                                                       metric_weights=self.optimization_goal)

    def _observation(self):
        """Return an observation of the env's current state."""
        return _shared_solution_to_observation(solution=self.solution, optimization_goal=self.optimization_goal,
                                               include_waldo_layers=False)

    def update_solution(self, action):
        """Given an action, reset and fill in the solution."""
        # TODO: Handle bonder priorities. For now not giving the agent control over them to simplify the action space.

        # Construct an export string based on the action array. This allows schem to handle all validity checks for us,
        # e.g. ensuring the correct number and types of features are placed without reimplementing schem's logic.

        # Template our export off of that exported by an empty solution to the level.
        # This gives us stuff like the correct name of reactor components for free.
        sample_soln = schem.Solution(None, level=self.level)
        sample_soln.author = 'Agent'
        # Grab the SOLUTION: and COMPONENT: lines
        export_str = '\n'.join(sample_soln.export_str().split('\n', maxsplit=2)[:2])

        # Add features
        for r, row in enumerate(action['features']):
            for c, feature_one_hot in enumerate(row):
                feature_idx = np.argmax(feature_one_hot)
                if feature_idx == 0:
                    continue

                feature_name = 'bonder' if features[1] and features[2] else FEATURE_IDX_TO_NAME[feature_idx - 1]
                export_str += f"\nMEMBER:'feature-{feature_name}',-1,0,1,{c},{r},0,0"

        # Add waldo instructions (note that by convention SC puts Start instrs before features, but schem doesn't care)
        for waldo, grid in enumerate(action['waldo_commands']):
            for r, row in enumerate(grid):
                for c, instr_type_one_hot in enumerate(row):
                    instr_type_idx = np.argmax(instr_type_one_hot)
                    if instr_type_idx == 0:
                        continue

                    # We can take advantage of schem's built-ins to generate export strings too
                    instr_type = INSTR_IDX_TO_TYPE[instr_type_idx - 1]
                    dirn_idx = np.argmax(action['waldo_command_directions'][waldo][r][c])
                    # Used by input/output and sense.
                    target_idx = (action['waldo_command_elements'][waldo][r][c]
                                  if instr_type == schem.waldo.InstructionType.SENSE
                                  else dirn_idx % 2)
                    instr = schem.waldo.Instruction(type=instr_type,
                                                    direction=CARDINAL_DIRECTIONS[dirn_idx],
                                                    target_idx=target_idx)
                    export_str += '\n' + instr.export_str(waldo_idx=waldo, posn=schem.grid.Position(col=c, row=r))

        # Add arrows
        for waldo, grid in enumerate(action['waldo_arrows']):
            for r, row in enumerate(grid):
                for c, dirn_one_hot in enumerate(row):
                    dirn_idx = np.argmax(dirn_one_hot)  # NONE, UP, RIGHT, DOWN, LEFT
                    if dirn_idx == 0:
                        continue

                    waldo_int = 64 if waldo == 0 else 16
                    dirn_degrees = (dirn_idx - 2) * 90  # SC uses -90 for UP through 180 for LEFT
                    export_str += f"\nMEMBER:'instr-arrow',{dirn_degrees},0,{waldo_int},{c},{r},0,0"

        # Pipe lines are optional in research levels

        self.solution = schem.Solution(export_str, level=self.level)


    def step(self, action):
        """This env has only a single step, taking all features / waldo symbols at once and running the solution.

        Reward is -2 for invalid action, -1 for non-solution, and the inverse fraction of the best human score for a
        successful solution (if the optimization goal mixes multiple metrics, the human best for each individual metric
        is used, so a score of 1 will likely be unachievable).

        Returns (state, reward, terminated, truncated, info); terminated is always True since this is a single-step env.
        """
        info = {}

        try:
            self.update_solution(action)
            score = self.solution.run(debug=schem.solution.DebugOptions(show_instructions=True))

            # Weighted sum as compared to human best (inverse since weighted score is being minimized)
            reward = self.top_human_score_weighted / sum(w * metric for w, metric in zip(self.optimization_goal, score))
        except SolutionImportError as e:
            reward = -2
            info['error'] = str(e)
        except SolutionRunError as e:
            reward = -1
            info['error'] = str(e)
        # Any other exception would be a bug in the simulator and worth raising

        # TODO: Observations should be for every cycle...
        return self._observation(), reward, True, False, info

    def reset(self):
        # TODO: Change levels if the agent beat the current level? If it reached X% of human best?
        #       If it *asks* us to via a separate action control?
        self.solution = schem.Solution(None, level=self.Level)
        obs = self._observation(self.solution, include_waldo_layers=True)
        info = {}

        return obs, info

    def render(self):
        """Pretty-print the current reactor state human-readably."""
        rich.print(self.solution.reactors)


class SChemEnvStepWise(gym.Env):
    """An SChem env where the input is the current partially-contructed solution, and the agent must place or remove
    another symbol or feature. The solution will be run after every change, returning a reward of 0 if the solution
    does not complete or has not improved its best score, and otherwise returning the difference in how much the
    solution has improved since its highest score ever reached.

    A closer simulation of how players 'actually' play the game, tweaking a solution until it works or to improve it.
    However, episodes may never end which would be tricky to train with.
    """
    pass


class SchemEnvSolutionString(gym.Env):
    """An SChem env where each action is a solution string as exported by SC CE."""
    pass
