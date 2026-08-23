from .constants import Constants
from .protein import Protein
from .cosolute import Cosolute
from .binary import CrowdingModel
from .statistics import monte_carlo_subsampling
from .plotting import BinaryPlotter

# Aliases for backwards compatibility with notebooks
var = Constants
protein = Protein
cosolute = Cosolute
crowding = CrowdingModel

__all__ = [
    "Constants",
    "Protein",
    "Cosolute",
    "CrowdingModel",
    "BinaryPlotter",
    "monte_carlo_subsampling",
    "var",
    "protein",
    "cosolute",
    "crowding"
]
