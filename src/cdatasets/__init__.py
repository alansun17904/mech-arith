from .arith_dataset import ArithDataset
from .bool_dataset import BooleanDataset
from .csense_dataset import CommonSenseDataset
from .dyck_dataset import DyckDataset
from .date_dataset import DateDataset
from .sports_dataset import SportsDataset
from .movie_dataset import MovieDataset
from .prompt_dataaset import PromptDataset
from .prompts import PromptFormatter, ZeroShot, FewShot, ChainOfThought


class DatasetBuilder:
    ids = {
        "arith": ArithDataset,
        "bool": BooleanDataset,
        "csense": CommonSenseDataset,
        "dyck": DyckDataset,
        "date": DateDataset,
        "sports": SportsDataset,
        "movie": MovieDataset,
    }

    def __init__(self, name):
        self.cls = self.ids[name]
        self.params = {}

    def set_param(self, name, val):
        self.params[name] = val
        return self

    def build(self):
        return self.cls(**self.params)


def get_dataset_strategy(name, **kwargs):
    return DatasetBuilder(name)


def get_prompt_strategy(name, **kwargs):
    if name == "zero-shot":
        return ZeroShot()
    elif name == "few-shot":
        return FewShot(kwargs["shots"])
    elif name == "chain-of-thought":
        return ChainOfThought()
    raise ValueError(
        f"Unknown prompt strategy: {name}; must be one of\
            'zero-shot', 'few-shot', 'chain-of-thought'."
    )


DatasetBuilder.get_strategy = get_dataset_strategy
PromptFormatter.get_strategy = get_prompt_strategy
PromptFormatter.ids = {
    "zero-shot": ZeroShot,
    "few-shot": FewShot,
    "chain-of-thought": ChainOfThought,
}
