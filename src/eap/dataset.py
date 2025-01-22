import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer


def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, list(labels)


class EAPDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """Creates a new EAPDataset object. Expects df to have
        columns "clean", "corrupted", and "label"
        """
        self.df = df

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["clean"], row["corrupted"], row["label"]

    def to_dataloader(self, batch_size: int, collate_fn=collate_EAP):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)
