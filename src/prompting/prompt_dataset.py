"""prompt_dataset.py
Generates a dataset of logical reasoning or factual recall tasks
and appends either chain of thought or few shot prompting to them.
"""


import json
import random
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import numpy as np
from torch.utils.data import Dataset


class TaskDataset(Enum):
    DYCK = (1, "dyck_languages")
    SPORTS = (2, "movie_recommendation")
    MOVIE = (3, "sports_understanding")
    DATE = (4, "date_understanding")
    ARITHMETIC = (5, "arithmetic")
    COMMON = (6, "common_sense")

    def __init__(self, id, path):
        self._id = id
        self._path = path

    @property
    def id(self):
        return self._id

    @property
    def path(self):
        return self._path
