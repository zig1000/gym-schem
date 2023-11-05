import copy
import math
import time

import rich

from .envs.schem_env import SChemEnv

def main():
    """As a demo, continuously runs random rollouts (via sample()) of the just-in-time env, until the user exits with
    Ctrl-C. Reports the highest reward solution found.
    """
    start = time.time()
    max_reward = -math.inf
    best_soln = None
    best_info = None
    eps = 0
    env = SChemEnv()
    try:
        while True:
            terminated = False
            while not terminated:
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            if reward > max_reward:
                max_reward = reward
                best_soln = copy.deepcopy(env.solution)  # reset() resets the solution so need to copy it
                best_info = info

            eps += 1
            env.reset()
    except KeyboardInterrupt:
        print(f"Best result after {eps} episodes:\n")
        print(best_soln.export_str() + '\n')
        # Pretty-print the best solution
        rich.print(next(best_soln.reactors).__str__(show_instructions=True, flash_features=False))
        if 'error' in best_info:
            print(best_info['error'])
        print(f"Reward: {max_reward}")

    print(f"Elapsed time: {time.time() - start:.1f}s")

if __name__ == '__main__':
    main()
