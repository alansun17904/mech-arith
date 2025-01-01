# Circuit Discovery through Patching: Choices and Best Practices
Herein, we detail exactly how patching is done, the choices taht we need to make
when we want to do circuit discovery and the best practices summarized by those
in the field. The purpose of this document is to clearly enumerate these choices/
hyperparameters so that in our paper we may explain them clearly and ablate over them.

At the core of these methodologies we care about three different metrics: faithfulness,
completeness, and simplicity [^2].

**Definition (Faithfulness).** The *faithfulness* of a circuit is the extent to which
they encapsulate the full model's computation on a particular task [^1].

## Ablation Methodologies
At the core of patching, we have control over the following:
1. the **granularity** of the computational grpah used to represent the model.
2. what type of **component** in the graph is ablated.
3. what type of **activation value** is used to ablate the component
4. which **token positions** are being ablated
5. the ablation **direction** (whether the ablation destroys or restores the signal)
6. the **set** of components ablated
Thus, a circuit-based ablation method is a six-tuple. And in our paper, for each
experiment we need to clearly specify our choices with respect to these six parameters.

It might not be advantageous to always perform mean ablation as is often done in the
literature. For example, consider a circuit for IOI that always uses a specific neuron
in the positional encoding. If we perform mean ablation, this neuron would not be apart
of the final circuit. On the other hand, if we perform zero ablation this would be a part
of the discovered circuit [^1]. Thus, this design decision should be something that we
are ablating over. 

## Handling Multiple Predicted Tokens
Much of the literature in activation patching is concerned with single token patching.
That is, the solution to the task is often a single token. However, in many of the tasks
that we are interested in this is not the case. 

Throughout our paper, when handling this case, we will use the following algorithm. First
assume that our task requires us to output two tokens $$t_1t_2$$ given input $$p$$. By
induction, we can derive the patching rule for more tokens. Let $$\mathbb{P}$$ be the
probability distribution over all possible output tokens $$t_1t_2$$. Also, let $a$
be the component in the neural network that we are ablating (specifically with value
$$\hat a$$). Then, we proceed as follows:

1. Compute $$\mathbb{P}(t_1 = t | \text{do}(a = \hat a))$$ using activation patching
on a single token. 
2. Let $$t$$ be the correct answer token for $$t_1$$. Then, compute 
$$\mathbb{P}(t_2 | t_1 = t, \text{do}(a=\hat a))$$. 
3. If $$t'$$ is the correct answer token for $$t_2$$, then by chain rule we have
that $$\mathbb{P}(t_1 = t, t_2 = t' | do(a = \hat a)) = 
\mathbb{P}(t_2 = t' | t_1 = t, \text{do}(a=\hat a))
\mathbb{P}(t_1 = t | \text{do}(a=\hat a))$$.

## Verifying Faithfulness
We will plot the Pareto frontier based on our patching results. Specifically,
we will:

1. Get the important of every edge/node through patching.
2. Sort the importance scores from most important to least important.
3. Create a line plot where the $$x$$-axis is the rank of how important that 
node was (so, max along the $$x$$-axis should be the total number of components
we are interested in patching) and the $$y$$-axis value that corresponds to a
specific $$x$$ is the amount of performance recovered after removing everything
accept for all $$x-1$$ components.
4. Perform a permutation test by randomly removing components and plotting the same
Pareto frontier. 


[^1]: Miller, Joseph, Bilal Chughtai, and William Saunders. "Transformer Circuit Faithfulness Metrics are not Robust." arXiv preprint arXiv:2407.08734 (2024).

[^2]: Wang, Kevin, et al. "Interpretability in the wild: a circuit for indirect object identification in gpt-2 small." arXiv preprint arXiv:2211.00593 (2022).