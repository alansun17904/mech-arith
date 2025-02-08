import json
import random
import pickle
from pathlib import Path
from functools import partial

from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate

from torch.utils.data import DataLoader


class PromptDataset(BaseDataset):
    """Investigating prompting stability"""
    
    def __init__(self, fname, n=1000):
        super().__init__()
        self.n = n
        self.fname = fname
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []
    
    def get_questions(self):
        task = pickle.load(open(self.fname, "rb"))
        for i in range(len(task["output"])):
            self._examples.append({"input": task["input"][i], "output": task["output"][i]})

        random.shuffle(self._examples)
        self._examples = self._examples[: self.n]

    def format_questions(self, formatter: PromptFormatter = None):
        if formatter is not None:
            raise ValueError("PromptDataset does not support any formatting.")
        self._clean_examples = [v["output"] for v in self._examples]
        self._corrupted_examples = [v["input"] for v in self._examples]
        self._labels = [v["output"] for v in self._examples]

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._clean_examples[idx], self._corrupted_examples[idx], self._labels[idx]
    
    def to_dataloader(self, model, batch_size: int, collate_fn=None):
        collate_fn = partial(generic_collate, model)
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)
        