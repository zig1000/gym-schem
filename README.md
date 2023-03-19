# gym-schem

An OpenAI env for the game SpaceChem.

## Why is SpaceChem a good research problem for ML?

* It's like programming, without requiring natural language to define the problems.
* SpaceChem has been demonstrate to be turing complete.
* It's like Sokoban on steroids
* Combinatorial search space
* Planning problem
* Multiple metrics to measure and/or optimize solutions by
* Solving levels is difficult (sparse rewards)
* Unlike Sokoban, levels have many non-trivially different approaches to solve them.
  While not having played Sokoban much, my impression of it is that sub-optimal
  Sokoban solutions are often equivalent to the optimal solution with extra non-functional
  walking around added or trivial swaps in the order boxes are pushed, but the same 'overall plan'.
  SpaceChem exhibits a combinatorially large number of meaningfully different approaches to solve
  any given level, with large variations in the resulting three metric scores of a solution.
* Well-plumbed dataset of optimal human scores for benchmarking quality of agent solutions.
  Optimal solutions for each of the three metrics are well explored by the playerbase for the
  current dataset of puzzles. Solutions spanning the pareto frontier of the three metrics
  have also been mapped by the playerbase to a lesser extent.
* Matching human scores requires a depth of planning and reasoning that in most levels likely can't be
  achieved by simply iteratively improving the same approaches that lead to basic solutions.
  This is an important product of the breadth of the solution space, and why I expect this
  environment to remain hard to match humans on well after superhuman Sokoban performance has been
  achieved.
* Curiosity and intuition are somewhat quantifiable. If an agent were left with the goal of matching/exceeding
  human scores (without being told what those scores are), it would behoove it to develop an intuition for which
  levels it still has room to improve its score on and should spend more time exploring, lest it continue
  fruitlessly attempting to improve a solution it had already perfected.
* Requires similar skills as programming or circuit design, but is much easier to quantify
  success in and is completely amenable to self-play.
  And unlike a lot of programming datasets, doesn't require an entire natural language processing
  ability to approach the problems.
* Automated curriculum-ifying of levels should be somewhat possible (e.g., simplifying the chemical conversion formula to get simpler inputs/outputs to work with).

## Why shouldn't I pick this environment?

* It's too sparse/difficult.
* If you don't feel Sokoban has been solved to a sufficient degree yet, you could use that.
* The input and output space is large; solutions consist of typically 10-100+ instructions, each of which
  is any of a dozen different instruction types and each placed anywhere in an 8x10 grid.
* Human solutions dataset is small. While optimal solutions are well-explored, only the top handful of solutons
  for each level have been preserved, with the path through the solution space that lead to each not preserved.
  This shortage of data could be improved in future.
* The dataset of levels is small. There are less than 500 preset 'intelligently-designed' puzzles. While
  procedural level generation is possible, these will have no human data and may be 'uninteresting' and/or
  too much easier or harder to solve than human-designed puzzles.
