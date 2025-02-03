import random

from torch.utils.data import Dataset
from transformer_lens.utils import get_attention_mask


COT = """\
Determine whether an artificially constructed sentence relating to sports is plausible or not.

Q: Is the following sentence plausible? "Bam Adebayo scored a reverse layup in the Western Conference Finals."
A: Let's think step by step. Bam Adebayo is an American basketball player. Scoring a reverse layup in the Western Conference Finals is part of the NBA Finals. So the answer is yes.

Q: Is the following sentence plausible? "Santi Cazorla scored a touchdown."
A: Let's think step by step. Santi Cazorla is a soccer player. Touchdown is part of American football and rugby. So the answer is no.

Q: Is the following sentence plausible? "DeMar DeRozan was called for the goal tend."
A: Let's think step by step. DeMar DeRozan is an American basketball player. Goal tending is part of basketball. So the answer is yes.
"""


class SportsDataset(Dataset):
    """A Dyck language task of variable difficulty."""

    def __init__(
        self, open_brackets="([{", closed_brackets=")]}", n=1000, max_length=15
    ):
        self.open_brackets = open_brackets
        self.closed_brackets = closed_brackets
        self.max_length = max_length
        self._closed_dict = {
            o: c for c, o in zip(self.closed_brackets, self.open_brackets)
        }
        self._probs = []
        for _ in range(n):
            self._probs.append(self._single_dyck())

    def _single_dyck(self):
        stack = []
        dyck = random.choice(self.open_brackets)
        stack.append(self._closed_dict[dyck])

        while len(dyck) + len(stack) < self.max_length:
            if len(stack) == 0 or random.random() < 0.5:
                o = random.choice(self.open_brackets)
                dyck += o
                stack.append(self._closed_dict[o])
            else:
                dyck += stack.pop()
        dyck += "".join(stack[::-1])
        return dyck[:-1], dyck[-1]

    @classmethod
    def _single_prompt(cls, dyck_q, dyck_a, cot=False, ans=True):
        header = "Correctly complete the following Dyck language string.\n"
        prob_header = "Q: Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: "
        if not cot:
            if ans:
                return header + f"{prob_header}{dyck_q}\nA: {dyck_a}\n"
            return header + f"{prob_header}{dyck_q}\nA: "
        return (
            COT
            + header
            + "\n"
            + f"{prob_header}{dyck_q}\nA: Let's think step by step.\n"
        )

    def to_str(self, shots=0, cot=False):
        """Converts Dyck language completions to their input
        prompts."""

        if cot:
            shots = 0

        prompts = []
        labels = []
        weights = [1 for _ in range(len(self._probs))]

        for i, prob in enumerate(self._probs):
            if shots:
                weights[i] = 0
                few_shots = random.choices(self._probs, weights=weights, k=shots)
                few_shot_header = "\n".join(
                    self._single_prompt(v[0], v[1], cot=False, ans=True)
                    for v in few_shots
                )
                few_shot_header += "\n"
                weights[i] = 1
            else:
                few_shot_header = ""

            prompt = self._single_prompt(prob[0], prob[1], cot=cot, ans=False)
            prompt = few_shot_header + prompt
            prompts.append(prompt)
        self.prompts = prompts

    def tok_probs(self, model):
        self.tokens = model.to_tokens(
            self.prompts, prepend_bos=True, padding_side="left"
        )
        self.attention_mask = get_attention_mask(self.tokens)
        return self.tokens, self.attention_mask

    def __len__(self):
        assert self.tokens is not None, "No tokens initialized in dataset"
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.attention_mask[idx], self.labels[idx]
