import pytest
import numpy as np
from transformers import AutoTokenizer


from src.arith_dataset import Op, ArithDataset


@pytest.fixture
def add_dataset():
	return ArithDataset(Op.ADD)

@pytest.fixture
def sub_dataset():
	return ArithDataset(Op.SUB)

@pytest.fixture
def mul_dataset():
	return ArithDataset(Op.MUL)

@pytest.fixture
def div_dataset():
	return ArithDataset(Op.DIV)

@pytest.fixture
def gpt_tokenizer():
	return AutoTokenizer.from_pretrained("gpt2")

@pytest.fixture
def llama_tokenizer():
	return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

@pytest.fixture
def gemma_tokenizer():
	return AutoTokenizer.from_pretrained("google/gemma-2-2b")

def test_op_enum():
	assert len(Op) == 4
	assert Op["ADD"].value == 1
	assert Op["SUB"].value == 2
	assert Op["MUL"].value == 3
	assert Op["DIV"].value == 4

def assert_num_digits(dig1, dig2, prob):
	"""Checks whether an arithmetic problem in the form of (op1, op2, ans)
	satisfies the constraint of op1, op2, having dig1, dig2 digits,
	respectively.

	Args:
		dig1 (int) 
		dig2 (int)
		prob (Tuple[int])
	Raises:
		AssertionError: if op1, op1 do not have dig1, dig2 digits.
	"""
	assert all([
		len(str(prob[0])) == dig1,
		len(str(prob[1])) == dig2
	])

def test_generate_add_dataset(add_dataset):
	for dig1 in range(1,11):
		for dig2 in range(1,11):
			## generate n=10 problems ##
			probs = add_dataset.arith_probs(dig1, dig2, 10)
			for prob in probs:
				assert_num_digits(dig1, dig2, prob)
				assert prob[0] + prob[1] == prob[2]
	assert add_dataset.problems is not None

def test_generate_sub_dataset(sub_dataset):
	for dig1 in range(1,11):
		for dig2 in range(1,11):
			## generate n=10 problems ##
			probs = sub_dataset.arith_probs(dig1, dig2, 10)
			for prob in probs:
				assert_num_digits(dig1, dig2, prob)
				assert prob[0] - prob[1] == prob[2]
	assert sub_dataset.problems is not None

def test_generate_mul_dataset(mul_dataset):
	for dig1 in range(1,11):
		for dig2 in range(1, 11):
			## generate n=10 problems ##
			probs = mul_dataset.arith_probs(dig1, dig2, 10)
			for prob in probs:
				assert_num_digits(dig1, dig2, prob)
				assert prob[0] * prob[1] == prob[2]
	assert mul_dataset.problems is not None

def test_generate_div_dataset(div_dataset):
	for dig1 in range(1, 11):
		for dig2 in range(1, dig1+1):
			## generate n=10 problems ##
			probs = div_dataset.arith_probs(dig1, dig2, 10)
			for prob in probs:
				assert_num_digits(dig1, dig2, prob)
				assert prob[0] / prob[1] == prob[2]
	assert div_dataset.problems is not None

def test_to_str_zero_shot(add_dataset):
	add_dataset.arith_probs(5, 3, n=10)
	assert add_dataset.problems is not None
	prompts = add_dataset.to_str(shots=0)
	assert add_dataset.prompts is not None
	assert len(prompts) == 10

	for i, prompt in enumerate(prompts):
		## check spacing ##
		assert len(prompt.split("\n")) == 1

		## check formatting ##
		assert int(prompt[:5]) == add_dataset.problems[i][0]
		assert int(prompt[8:11]) == add_dataset.problems[i][1]

def test_to_str_few_shot(add_dataset):
	add_dataset.arith_probs(3,2,n=15)
	assert add_dataset.problems is not None
	prompts = add_dataset.to_str(shots=15)
	assert add_dataset.prompts is not None
	assert len(prompts) == 15

	for i, prompt in enumerate(prompts):
		## check all shots are accounted for ##
		assert len(prompt.split("\n")) == 16
		
		## check no overlaps between prompt and shots ##
		prompt_lines = prompt.split("\n")
		shots = prompt_lines[:15]
		quest = prompt_lines[-1]
		assert not any([
			v[:-3] == quest for v in shots
		])

def test_gpt_tok_probs(add_dataset, gpt_tokenizer):
	add_dataset.arith_probs(4, 3, n=100)
	add_dataset.to_str()
	gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
	add_dataset.tok_probs(gpt_tokenizer)
	assert len(add_dataset) == 100

def test_llama_tok_probs(add_dataset, llama_tokenizer):
	add_dataset.arith_probs(4, 3, n=100)
	add_dataset.to_str()
	llama_tokenizer.pad_token = llama_tokenizer.eos_token
	add_dataset.tok_probs(llama_tokenizer)
	assert len(add_dataset) == 100

def test_gemma_tok_probs(add_dataset, gemma_tokenizer):
	add_dataset.arith_probs(4, 3, n=100)
	add_dataset.to_str()
	gemma_tokenizer.pad_token = gemma_tokenizer.bos_token
	add_dataset.tok_probs(gemma_tokenizer)
	assert len(add_dataset) == 100

def test_parse_ans(add_dataset, sub_dataset, mul_dataset, div_dataset):
	op1, op2, ans = add_dataset.parse_ans("2384 + 238521 = 48523109")
	assert op1 == 2384
	assert op2 == 238521
	assert ans == 48523109

	op1, op2, ans = sub_dataset.parse_ans("2384 - 238521 = 48523109")
	assert op1 == 2384
	assert op2 == 238521
	assert ans == 48523109

	op1, op2, ans = mul_dataset.parse_ans("2384 * 238521 = 48523109")
	assert op1 == 2384
	assert op2 == 238521
	assert ans == 48523109

	op1, op2, ans = div_dataset.parse_ans("2384 / 238521 = 48523109")
	assert op1 == 2384
	assert op2 == 238521
	assert ans == 48523109

	op1, op2, ans = add_dataset.parse_ans("24 + 429 = abc")
	assert op1 == 24
	assert op2 == 429
	assert ans == -np.inf

def test_score_add(add_dataset):
	a1 = """
	45 + 7 = 412
	582 + 592 = 1
	34 + 35 = 69
	"""
	a2 = """
	230 + 28 = 1
	"""
	a3 = """
	28 + 459 = absc
	"""
	assert add_dataset.score(a1) == 1
	assert add_dataset.score(a2) == 0
	assert add_dataset.score(a3) == 0
	
def test_score_sub(sub_dataset):
    """Tests the score function for subtraction problems."""
    s1 = """
    10 - 5 = 5
    20 - 12 = 8
    100 - 30 = 70
    """
    s2 = """
    15 - 8 = 5
    50 - 25 = 24
    """
    assert sub_dataset.score(s1) == 1
    assert sub_dataset.score(s2) == 0

    ## Test with negative results ##
    s3 = """
    5 - 10 = -5
    20 - 30 = -10
    """
    assert sub_dataset.score(s3) == 1 

    ## Test with edge cases ##
    s4 = """
    0 - 0 = 0
    1 - 1 = 0
    """
    assert sub_dataset.score(s4) == 1

def test_score_mul(mul_dataset):
    """Tests the score function for multiplication problems."""
    m1 = """
    2 * 3 = 6
    5 * 4 = 20
    10 * 10 = 100
    """
    m2 = """
    3 * 3 = 8
    2 * 5 = 11
    """
    assert mul_dataset.score(m1) == 1
    assert mul_dataset.score(m2) == 0

    ## Test with zero ##
    m3 = """
    0 * 5 = 0
    5 * 0 = 0
    """
    assert mul_dataset.score(m3) == 1

def test_score_div(div_dataset):
    """Tests the score function for division problems."""
    d1 = """
    10 / 2 = 5
    15 / 3 = 5
    20 / 4 = 5
    """
    d2 = """
    12 / 4 = 2
    10 / 5 = 3 
    """
    assert div_dataset.score(d1) == 1
    assert div_dataset.score(d2) == 0

    ## Test with remainders ## 
    d3 = """
    7 / 2 = 3.5 
    11 / 3 = 3.66
    """
    assert div_dataset.score(d3) == 0