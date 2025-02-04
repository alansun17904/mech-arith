"""prompt_dataset.py
Generates a dataset of logical reasoning or factual recall tasks
and appends either chain of thought or few shot prompting to them.
"""


import json
import random
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from torch.utils.data import Dataset


class TaskDataset(Enum):
    DYCK = (1, "dyck_languages")
    SPORTS = (2, "movie_recommendation")
    MOVIE = (3, "sports_understanding")

    def __init__(self, id, path):
        self._id = id
        self._path = path

    @property
    def id(self):
        return self._id

    @property
    def path(self):
        return self._path


class PromptDataset(Dataset):
    """Dataset of various tasks with different modes of prompting.
    Generates problems from a given task with any modes of prompting:
    [Zero-shot, Few-shot, or Chain-of-thought]."""

    def to_str(self, shots: int = 0):
        ...
