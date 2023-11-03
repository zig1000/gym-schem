"""Helper module for fetching top human scores by level"""

import math
import requests
from typing import Union

import schem

ARCHIVE_BASE_URL = 'https://raw.githubusercontent.com/spacechem-community-developers/spacechem-archive/master'

# Use the archive README's list of level -> folder links (yes, I know this is Bad) to construct a dict of level names
# to their relative paths in the archive. Note that duplicate levels have their resnet ID suffixed, a behaviour we'll
# also borrow as otherwise distinguishing duplicate level names is non-trivial.
level_name_to_archive_path = {}
readme_text = requests.get(f'{ARCHIVE_BASE_URL}/README.md').text
for line in readme_text.split('\n'):
    if not line.startswith('['):
        continue

    parts = line.strip()[1:-1].split('](')  # Safely parses "[level_name](path)  "
    if len(parts) != 2:
        continue

    level_name, archive_path = parts
    # Filter out any random links that aren't levels, accounting for duplicated level names having resnet ID appended.
    if level_name not in schem.levels and level_name.rsplit(maxsplit=1)[0] not in schem.levels:
        continue

    level_name_to_archive_path[level_name] = archive_path

def get_level_scores_raw(level_name: str):
    """Given a level name, return the raw text in its archive README page, which lists all pareto scores w/ tags."""
    if level_name not in level_name_to_archive_path:
        if level_name in schem.levels:  # In which case it must be a duplicate level
            raise ValueError(f"Level name `{level_name}` is ambiguous; add resnet ID, e.g. `{level_name} (1-2-3)`.")

        raise ValueError(f"Invalid level name `{level_name}`.")

    archive_url = f'{ARCHIVE_BASE_URL}/{level_name_to_archive_path[level_name]}/solutions.psv'
    return requests.get(archive_url).text

def top_human_score_weighted(level_name: str, metric_weights: tuple[float, float, float]):
    """Given a level and weighted metric target (e.g. (0.99, 0, 0.01)), return the pareto frontier score that
    minimizes the weighted sum, and the resulting sum.
    """
    top_scores = get_level_scores_raw(level_name)  # Raw README text with scores and tags
    min_score = math.inf
    best_sc_score = None
    for line in top_scores.split('\n'):
        if not line:
            continue

        score_str = line.split('|', maxsplit=1)[0]
        score_parts = score_str.split('/')
        # Skip bugged solutions (but not precog solutions)
        if len(score_parts) == 4 and 'B' in score_parts[-1]:
            continue

        sc_score = schem.Score(*(int(x) for x in score_parts[:3]))
        score = sum(weight * metric for weight, metric in zip(sc_score, metric_weights))
        if score < min_score:
            min_score = score
            best_sc_score = sc_score

    return best_sc_score, min_score

metric_prefix_to_idx = {'C': 0, 'R': 1, 'S': 2}
def top_human_score_by_tag(level_name: str, metric: Union[str, int]):
    """Get the top score for the given level in the given metric.

    If the level is one of the duplicate levels (Pyridine, Breakdown, Vitamin B3), the given level name must also be
    suffixed with the resnet ID, separated by a space, e.g. "Pyridine (2-1-1)".

    `metric` can be either:
    - The tag used by https://github.com/spacechem-community-developers/spacechem-archive for labelling top solutions by
      category (should always include NB ("No Bugs") as schem doesn't support legacy bugs mode).
      Allowed values: 'CNB', 'CNBP', 'SNB', 'SNBP', 'RCNB', 'RSNB', 'RCNBP', 'RSNBP'.
    - A 0-2 index, which per the Score order, uses CNB, RCNB, and SNB respectively.
    """
    # If given a metric index instead of the archive's expected tag format, convert
    if isinstance(metric, int):
        assert 0 <= metric <= 2
        metric = 'CNB' if metric == 0 else ('RCNB' if metric == 1 else 'SNB')
    else:
        assert 'NB' in metric, """Please target a tag containing "NB" as schem doesn't support legacy bugs."""

    top_scores = get_level_scores_raw(level_name)  # Raw README text with scores and tags
    for line in top_scores.split('\n'):
        if not line:
            continue

        score_str, *_, tags = line.split('|')
        if metric in tags.split(','):
            return schem.Score.from_str(score_str.replace('/', '-'))[metric_prefix_to_idx[metric[0]]]

    # Handle the silly case of reactor metric being measured on a research level (reactor tag will be missing)
    if metric[0] == 'R':
        return 1

    raise Exception(f"Could not find metric tag {metric} in {archive_url}")


if __name__ == '__main__':
    print(top_human_score_weighted("An Introduction to Bonding", metric_weights=(1, 0, 0)))
