import tqdm
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

access_token = "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"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=access_token
)

import os

# Detect the device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple M1/M2 GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
elif 'HSA_PATH' in os.environ or os.path.exists('/opt/rocm'):
    device = torch.device("cuda")  # Assume ROCm is available (PyTorch treats this as CUDA)
else:
    device = torch.device("cpu")   # Fallback to CPU

# Example usage
print(f"Using device: {device}")



class Toks(Dataset):
    def __init__(self, toks):
        self.toks = toks
    
    def __len__(self):
        return len(self.toks["input_ids"])
    
    def __getitem__(self, idx):
        return self.toks["input_ids"][idx], self.toks["attention_mask"][idx]


for i in range(1,7):
    for j in range(0, i+1):
        examples = open(f"{i}digit-{j}carry-problems+new.txt", "r").readlines()
        few_shot_examples = "100 + 200 = 300\n520 + 890 = 1410\n"
        a = [few_shot_examples + v.strip() + " " for v in examples]
        toked = tokenizer.batch_encode_plus(a, return_tensors="pt", padding=True)


    dl = DataLoader(Toks(toked), batch_size=32)
    texts = []
    for x, y in tqdm.tqdm(dl):
        x = x.to(device)
        y = y.to(device)
        
        # x = x.to("cuda")
        # y = y.to("cuda")
        outputs = model.generate(input_ids=x, attention_mask=y, max_new_tokens=32)
        texts.append(tokenizer.batch_decode(outputs))

    pickle.dump(texts, open(f"{i}digit-{j}carry-results+new.pkl", "wb"))




