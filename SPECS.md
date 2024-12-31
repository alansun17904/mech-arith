# Code Specifications
We detail the desired experiments for the paper as well as the code specifications to implement those experiments.
We need three key modules: dataset processing and loading, network patching, and an evaluation
module to run patching at scale. 

The source code for the main modules can be found in `src/` their unit tests are in `tests/` and quick
explorations as well as generation of the papers main figures can be found in the `notebooks/` directory.

When constructing these modules code quality is the most important. Every function needs to have 
docstrings (https://google.github.io/styleguide/pyguide.html). Moreover, every class and functions
need to have the appropriate unit tests. Even though this will make progress a little slow at first
it will help us ensure both the correctness and scalability of our experiments. 

## Unit Testing
Unit testing is done through `pytest`. Make sure that `pytest` is installed using `pip`. Then, simply
run `pytest .` in the root directory. 
