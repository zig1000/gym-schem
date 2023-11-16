from .envs import SChemEnv, SChemEnvOneShot
from .wrappers import FlattenAction
# TODO: When running `python -m gym_schem` this __main__ import causes a warning, but without it other modules can't
#       run this package's main() function. Am I seriously expected to have both a main.py and __main__.py to do this?
from .__main__ import main

__version__ = "0.1.0"

# TODO: Register per https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#registering-envs
