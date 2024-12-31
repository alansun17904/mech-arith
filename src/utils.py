import os
import math
import torch
import random
import transformers
import numpy as np


def seed_everything(seed: int=42):
	random.seed(seed)
	transformers.set_seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True