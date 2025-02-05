from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Dataset of various tasks with different modes of prompting.
    Generates problems from a given task with any modes of prompting:
    [Zero-shot, Few-shot, or Chain-of-thought]."""
    def __init__(self):
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []

    @property
    def examples(self):
        return self._examples

    @abstractmethod
    def get_questions(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...

    @abstractmethod
    def to_dataloader(self, batch_size: int, collate_fn):
        ...

    @abstractmethod
    def format_questions(self, formatter):
        ...
