"""arith_bench.py
Generates a dataset of random arithmetic problems of various lengths with
different operations and outputs them as a json file. 
and outputs them as a json file. 
"""


import re
import random
from enum import Enum
from typing import Tuple, List, Dict

import numpy as np
from torch.utils.data import Dataset


class Op(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4


class ArithDataset(Dataset):
    """Dataset of arithmetic problems. Generates problems with the specified
    number of digits on the fly with options for few-shot prompting.
    """
    op: Enum
    problems: List[Tuple[int]]
    prompts: List[str]
    operator = {
        Op.ADD: "+",
        Op.SUB: "-",
        Op.MUL: "*",
        Op.DIV: "/",
    }
    tokens: Dict

    def __init__(self, op: Op, probs=None, prompts=None, tokens=None):
        self.op = op
        self.problems = probs
        self.prompts = prompts
        self.tokens = tokens
        self.shots = 0

    def arith_probs(self, dig1: int, dig2: int, n=1000):
        """Generate binary arithmetic problems.

        Args:
            dig1 (int): the number of digits in the first operand.
            dig2 (int): the number of digits in the second operand.
            n (int): the total number of problems.
        Returns:
            a list of tuple of three integers (op1, op2, ans) where op1, op2
            have dig1, dig2 number of digits, respectively;
            and op1 +/- op2 = ans. Note that op1 >= 0 and op2 >= 0.
        """
        problems = []
        for _ in range(n):
            problems.append(self._one_prob(dig1, dig2))
        self.problems = problems
        return problems

    def _one_prob(self, dig1: int, dig2: int):
        """Generates a single arithmetic problem.

        Args:
            dig1 (int): the number of digits in the first operand.
            dig2 (int): the number of digits in the second operand.
        Returns:
            a tuple of three integers (op1, op2, ans), where
                op1 ... op2 = ans,
            where ... is the specified operation in the instantiated
            class.
        """
        if self.op is Op.DIV:
            assert (
                dig1 >= dig2
            ), "To ensure division results in integer \
            solutions operand 1 must have more digits than operand 2."
            op2 = random.randint(int(10 ** (dig2 - 1)), int(10 ** dig2 -1))
            ans = random.randint(int(10 ** (dig1 - dig2)), int(10 ** (dig1 - dig2 + 1) - 1))
            op1 = ans * op2

            if len(str(op1)) != dig1:
                ## shorten ans to account for op1 having more digits than dig1 ##
                ans = ans // 10 + 1
                op1 = ans * op2

            return (op1, op2, ans) 
        op1 = random.randint(int(10 ** (dig1 - 1)), int(10**dig1 - 1))
        op2 = random.randint(int(10 ** (dig2 - 1)), int(10**dig2 - 1))
        if self.op is Op.ADD:
            return (op1, op2, op1 + op2)
        elif self.op is Op.SUB:
            return (op1, op2, op1 - op2)
        else:
            return (op1, op2, op1 * op2)

    def to_str(self, probs: List[Tuple[int]]=None, shots: int=0):
        """Converts arithmetic problems to their input prompts.

        Args:
            probs (List[Tuple[int]]): list of arithmetic problems generated by
                `self.arith_probs`. 
            shots (int): Number of few-shot prompts to prepend to each input
                prompt. 
        Returns:
            List[str] of input prompts.
        Raises:
            AssertionError: if shots >= len(probs).
        """
        if probs is None and self.problems is None:
            raise ValueError("No problems provided. Either pass manually pass through \
                `probs` argument or run `arith_probs`.")
        probs = probs if probs is not None else self.problems
        prompts = []
        oper = self.operator[self.op]
        weights = [1 for _ in range(len(probs))]
        for i, prob in enumerate(probs):
            ## randomly sample shots number of examples and add to prompt ##
            weights[i] = 0
            few_shots = random.choices(probs, weights=weights, k=shots)
            few_shot_header = "\n".join(
                [f"{v[0]} {oper} {v[1]} = {v[2]}" for v in few_shots]
            )
            few_shot_header += "\n" if len(few_shot_header) > 0 else ""
            prompt = few_shot_header + f"{prob[0]} {oper} {prob[1]} = "
            weights[i] = 1
            prompts.append(prompt)
        self.prompts = prompts
        self.shots = shots
        return prompts

    def tok_probs(self, tokenizer, prompts=None):
        """Tokenize a list of prompts that have been generated by `self.to_str`

        Args:
            tokenizer: model tokenizer
            prompts (List[str]): input prompts passed to model
        Returns:
            tokenized prompts in the default HuggingFace format
        """
        if self.prompts is None and prompts is None:
            raise ValueError("No input prompts are given, either manually provide \
                them, or generate them using the `arith_probs` + `to_str` methods.")
        prompts = prompts if prompts is not None else self.prompts
        self.tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
        return self.tokens

    def __len__(self):
        assert self.tokens is not None, "No tokens initialized in dataset"
        return len(self.tokens["input_ids"])

    def __getitem__(self, idx):
        return self.tokens["input_ids"][idx], self.tokens["attention_mask"][idx]

    def parse_ans(self, result: str):
        """Given the causal response from a language model parse the problem
        it was required to solve using regex.

        Args:
            result (str): Resulting output from a LM.
        Returns:
            a tuple of three integers (a, b, ans) where the given question to
            LM was `a (op) b = ` and the LM's response was `ans`.
        """
        ## remove all few-shot examples ##
        print(result, "\n -----")
        response = result.split("\n")[self.shots-1]
        ptrn = "(\d+)\s\\" + self.operator[self.op] + r"\s(\d+)\s=\s(\d+)"
        print(response)
        srch = re.search(ptrn, response)
        if srch is not None:
            return (int(srch.group(1)), int(srch.group(2)), int(srch.group(3)))
        else:  ## model did not output numerical response ##
            ## retrieve problem and output inf ## 
            ptrn = r"(\d+)\s\\" + self.operator[self.op] + r"\s(\d+)\s="
            srch = re.search(ptrn, response)
            return (int(srch.group(1)), int(srch.group(2)), -np.inf)