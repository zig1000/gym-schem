# gym-schem

A [Gym](https://github.com/Farama-Foundation/Gymnasium) env for the programming puzzle game [SpaceChem](https://www.zachtronics.com/spacechem).

## Getting Started

The just-in-time env is an ideal starting point. It places instructions on the circuit just-in-time for each waldo
(controller) to execute them. This reduces the chance of placing 'dead' instructions that never execute, reduces the
action space to a manageable size, and more tightly ties each action to the observation of its effect on the environment
(since it gets executed within 1-2 steps).

Conversely, the one-shot env has only a single step, placing all instructions at once.

## Why is SpaceChem a good ML research problem?

*   It's basically programming, but without messy natural language involved.
*   It's 'deep': It has a sparse, combinatorial search space, and requires planning, reasoning, and creativity to
    match top human scores, in a way that could not simply be generalized from prior similar-looking levels.
    *   For example, one level's top reactor/symbol score implements Bogosort, something not useful in other levels.
        These wildly different top approaches are emergent from simple differences in levels' input and goal molecules.
*   A large and varied set of levels plus various metrics to optimize ensure an almost unbounded ceiling for self-play.  
    Basically once this env is solved I'll *actually* start being scared of AGI.
* Well-plumbed dataset of optimal human scores for benchmarking quality of agent solutions.
  Optimal solutions for each of the three metrics are well explored by the playerbase for the
  current dataset of puzzles. Solutions spanning the pareto frontier of the three metrics
  have also been mapped by the playerbase to a lesser extent.

## Why shouldn't I pick this environment?

*   Large observation size, on the order of ~5000 ints per step.
*   If you don't feel Sokoban has been solved to a sufficient degree yet, you could use that for its simplicity.
*   Human solutions dataset is small. While optimal solutions are well-explored, only the top handful of solutions
    for each level have been preserved, with the 'optimization path' not preserved.
    This shortage of data could be improved in future.

## Roadmap

*   Provide a utility for taking a level and automatically creating a 'curriculum' of levels aimed at improving agent
    generalization, e.g.:
    *   Vary input molecule positions
    *   Reduce required output count
    *   Reduce 'distance' between input and output molecules
*   Improve reward function's measurement of chemical 'distance' between env molecules and goal molecule(s)
*   Provide a utility for taking a solution export string (as exported by the Community Edition steam beta) and
    converting it to a list of actions, for the purpose of supervised training from human data
*   Support Production levels
*   Random level generation (tends to produce less interesting high-difficulty levels, but should be good for
    easy levels). Lower priority than curriculumifyng existing levels.
