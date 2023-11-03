import copy
from enum import IntEnum

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Box, Dict, Discrete, Tuple
import numpy as np
import rich

import schem
from schem import *
from schem.components import Reactor, RandomInput
from schem.grid import CARDINAL_DIRECTIONS, Direction, Position
from schem.waldo import Waldo, Instruction, InstructionType

from human_scores import top_human_score_weighted

ROTATIONAL_DIRECTIONS = (Direction.CLOCKWISE, Direction.COUNTER_CLOCKWISE)
NUM_WALDOS, NUM_COLS, NUM_ROWS = Reactor.NUM_WALDOS, Reactor.NUM_COLS, Reactor.NUM_ROWS
INSTR_IDX_TO_TYPE = list(schem.waldo.InstructionType)  # Mapping of instruction action indices to schem's native enum
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
    'waldo_command_elements': Box(shape=(NUM_WALDOS, NUM_ROWS, NUM_COLS), dtype=np.int16, low=1, high=203),
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
    obs = gym.vector.utils.numpy_utils.create_empty_array(shared_observation_dict)

    for i, metric_weight in enumerate(optimization_goal):
        obs['optimization_target'][i] = metric_weight

    # schem handles bonders differently since they mix priorities with plus vs minus bonders; store the correct type
    # (and store as both + and - for regular +- bonders).
    # For now I'm hiding feature priorities from the agent like OG SC.
    for (c, r), bond_type in solution.bonders:
        if '+' in bond_type:
            obs['features'][r][c][0] = 1
        if '-' in bond_type:
            obs['features'][r][c][1] = 1

    for i, feature in enumerate(('sensors', 'fusers', 'splitters', 'swappers')):
        for c, r in getattr(solution, feature):
            # Note that fusers and splitters store only their left position. This is probably preferable.
            obs['features'][r][c][i + 3] = 1

    for i, input in solution.inputs:
        for (c, r), atom in input.input_molecule.atom_map:
            obs['input_atoms'][i][r][c] = atom.element.atomic_num
            obs['input_atoms_max_bonds'][i][r][c] = atom.element.max_bonds
            obs['input_bonds'][i][r][c][0] = atom.bonds[RIGHT] if RIGHT in atom.bonds else 0
            obs['input_bonds'][i][r][c][1] = atom.bonds[DOWN] if DOWN in atom.bonds else 0

    for i, output in solution.outputs:
        obs['target_output_counts'][i] = output.count
        molecule = output.output_molecule
        for (c, r), atom in molecule.atom_map.items():
            obs['output_atoms'][i][r][c] = atom.element.atomic_num
            obs['output_atoms_max_bonds'][i][r][c] = atom.element.max_bonds
            obs['output_bonds'][i]
            obs['output_bonds'][i][r][c][0] = atom.bonds[RIGHT] if RIGHT in atom.bonds else 0
            obs['output_bonds'][i][r][c][1] = atom.bonds[DOWN] if DOWN in atom.bonds else 0

    reactor = solution.reactors[0]
    for molecule in reactor.molecules:
        for (c, r), atom in molecule.atom_map.items():
            obs['atoms'][r][c] = atom.element.atomic_num
            obs['atoms_max_bonds'][r][c] = atom.element.max_bonds
            obs['bonds'][r][c][0] = atom.bonds[RIGHT] if RIGHT in atom.bonds else 0
            obs['bonds'][i][r][c][1] = atom.bonds[DOWN] if DOWN in atom.bonds else 0

    if include_waldo_layers:
        for w, waldo in reactor.waldos:
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


# TODO: It is probably better to avoid this 'alternating actions' scheme and have each step() accept both an arrow
#       and an instruction, since neither can 'affect' the result of the other (insofar as order doesn't matter).
#       This way the only alternating state is the waldo being placed. We could merge the waldo steps too but then
#       we lose the 'symmetry' of the waldos and double the output size, not that it's large anyway.
#       PLUS if we merge all the action spaces, the 'waldo Start phase' is not special; we simply ignore the
#       instruction part of the input and read from the cell + direction parts.
class SChemEnvJustInTime(gym.Env):
    """An SChemEnv implementation in which solution symbols are placed just-in-time while the solution is being run.
    When revisiting a grid cell in which the agent was already given the opportunity to place a symbol (whether or not
    they did), that cycle is fast-forwarded past, with step() potentially simulating multiple cycles per step.

    Problem: fast-forwarding cycles means either skipping observations or a dynamic observation size. Could only show
             the new frames, accept being dynamic, or have steps where the given action is ignored.
             It'd be nice for this version of the env to be the one with static observation size though.

    In full, a run proceeds as follows:
    first N step() calls: Agent must select a (col, row) coordinate to place each feature in turn until all N features
                          (bonders, sensor, fuser, etc.) of the level are placed.
    Next 2 step() calls: Agent must select a coordinate for the current waldo's start location, along with the Start
                         command's direction and any arrow for the starting cell.
    Remaining step() calls: Agent alternates placement for each waldo (red then blue), placing up to one command and one
        arrow per step. After blue steps, cycle movement is performed.
        he observation space will include a value indicating which waldo the current step is for,
        and will skip a waldo's step() during a cycle if it is in a grid cell it has already visited
        (e.g. flip-flops can't be placed only on the second time a grid cell is
        visited, even though they wouldn't have affected the first pass).
        TODO: maybe allow the flip-flop case to make it easier for the agent to learn what they do?
    """
    metadata = {'render.modes': ['human']}
    last_level_solved = -1
    #max_training_level

    def __init__(self, optimization_goal=(0.99, 0, 0.01), render_mode=None):
        # Gymnasium API properties
        self.observation_space = Dict({**shared_observation_dict,
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
        self.action_space = Dict({
            # col/row only used during feature and waldo Start placement
            'col': Box(shape=(1,), dtype=np.int8, low=0, high=Reactor.NUM_COLS - 1),
            'row': Box(shape=(1,), dtype=np.int8, low=0, high=Reactor.NUM_ROWS - 1),
            'arrow': Discrete(5),  # NONE, UP, RIGHT, DOWN, LEFT
            'command': Discrete(len(InstructionType) - 2),  # 0 = no command, and we exclude START, CTRL, and PAUSE
            'command_direction': Discrete(4),  # Ignored for non-directional instructions. Used for waldo Start.
            # Box not Discrete since atomic number is in some sense 'ordered'?
            'command_element': Box(shape=(1,), dtype=np.int16, low=1, high=203)})  # Only used by sense.

        # Internal vars
        self.level = schem.Level(next(iter(schem.levels.values())))  # Of Pancakes and Spaceships
        self.solution = schem.Solution(None, level=self.level)
        self._reactor = next(self.solution.reactors)
        self.optimization_goal = optimization_goal
        self.top_human_score, self.top_human_score_weighted = top_human_score_weighted(self.level.name,
                                                                                       metric_weights=optimization_goal)

        # Check which waldo command action indices are illegal (will be ignored)
        self.disallowed_command_actions = set()
        instr_to_indices = {'instr-bond': {8, 9}, 'instr-sensor': {10}, 'instr-toggle': {11},
                            'instr-fuse': {12}, 'instr-split': {13}, 'instr-swap': {14},
                            'instr-control': set()}  # We exclude PAUSE and CTRL from the action space already
        for disallowed_instr in self._reactor.disallowed_instrs:
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
        obs['waldo_being_placed'] = self.active_waldo
        return obs

    def step(self, action):
        state, reward, terminated, truncated, info = [], 0, False, False, {}
        reactor = self._reactor

        if not self._placed_features:
            # Check how many features we have left to place

            self._placed_features = True
            # Re-init Reactor's bonder pair helper properties
            reactor.bond_plus_pairs, reactor.bond_minus_pairs = reactor.bond_pairs()

            return state, reward, terminated, truncated, info

        # Skip command/arrow placement if the current waldo has already visited this cell
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
                        else ROTATIONAL_DIRECTIONS[action['command_direction'] <= 1])  # UP, RIGHT -> CLOCKWISE

                # Used by input/output and sense.
                target_idx = (action['command_element']
                              if instr_type == InstructionType.SENSE
                              else int(action['command_direction'] <= 1))

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

            # Instant actions + check for completion. Bit of DRY violation with schem here.
            for component in self.solution.components:
                if (component.do_instant_actions(self.solution.cycle)  # True if it's an Output that just completed
                        and all(output.current_count >= output.target_count for output in self.solution.outputs)):
                    score = Score(self.cycle - 1, len(self.solution.reactors), self.symbols)
                    # Weighted sum as compared to human best (inverse since weighted score is being minimized)
                    reward = self.top_human_score_weighted / sum(w * m for w, m in zip(self.optimization_goal, score))
                    return state, reward, True, truncated, info

            self.solution.cycle_movement()

            if self.solution.does_it_halt():  # Detects infinite loops and fast-forwards cycles to completion
                score = Score(self.cycle - 1, len(self.solution.reactors), self.symbols)
                # Weighted sum as compared to human best (inverse since weighted score is being minimized)
                reward = self.top_human_score_weighted / sum(w * m for w, m in zip(self.optimization_goal, score))
                return state, reward, True, truncated, info

        self.active_waldo = 1 - self.active_waldo  # Flip waldo

        #state = self._observation()
        return state, reward, terminated, truncated, info

    def reset(self):
        # TODO: Change levels if the agent beat the current level?
        self.solution.reset()
        state = 0
        return state

    def render(self):
        """Pretty-print the current reactor state human-readably."""
        rich.print(next(self.solution.reactors).__str__(show_instructions=True, flash_features=False))


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
    However, episodes may never end which might be tricky to train with.
    """
    pass


class SchemEnvSolutionString(gym.Env):
    """An SChem env where each action is a solution string as exported by SC CE."""
    pass


if __name__ == '__main__':
    if False:
        # As a proof-of-concept, generate and run an import-valid solution via Space.sample().
        env = SChemEnvOneShot()
        sample_action = env.action_space.sample()

        # Give the fully-random action some help to become validly importable
        ## Zero the features since pancakes has none
        sample_action['features'] = np.zeros(env.action_space['features'].shape,
                                             dtype=env.action_space['features'].dtype)

        ## Zero all Start and illegal instrs, then re-randomize starts
        for waldo_grid in sample_action['waldo_commands']:
            for row in waldo_grid:
                for instr_one_hot in row:
                    instr_one_hot[1] = 0  # Zero out Start
                    for i in range(9, len(instr_one_hot)):  # Zero out bond instruction and above
                        instr_one_hot[i] = 0

        sample_action['waldo_commands'][0][np.random.randint(NUM_ROWS)][np.random.randint(NUM_COLS)][1] = 1
        sample_action['waldo_commands'][1][np.random.randint(NUM_ROWS)][np.random.randint(NUM_COLS)][1] = 1

        obs, reward, terminated, truncated, info = env.step(sample_action)
        if 'error' in info:
            print(info['error'])
    else:
        # As a proof-of-concept, generate and run an import-valid solution via Space.sample().
        env = SChemEnvJustInTime()
        try:
            while True:
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                if 'error' in info:
                    print(info['error'])
                    break
        finally:
            print('\n' + env.solution.export_str() + '\n')
            env.render()  # Print the final solution
