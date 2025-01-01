# Code Specifications
We detail the desired experiments for the paper as well as the code specifications to implement those experiments.

## Directory Structure
The source code for the main modules can be found in `src/` and their unit tests
are in `tests/`. Generally, we need `src/` and `test/` to mirror the same 
directory structure. So, if you have a folder `src/datasets` then the corresponding
folder also needs to exist in `test/`: `test/test_datasets`.

General explorations and figure generations using notebooks should all be contained
in `notebooks/`. Any code in these files don't need unit tests, but they should
only contain jupyter notebooks.

## Adding Code
When adding new code, a specific issue should be adressed. No one should be pushing
directly to main. We will follow this workflow

1. Find an issue that you want to address.
2. Create a new branch titled `n-[title]`, where `n` is replaced with the issue
number. For example, if I am working on issue #1: refactoring baseline, then my
branch name could be something like `1-refact-baseline`.
3. Checkout the branch and add your changes in this branch. Each commit to the
branch should reference the issue that you are tackling. Continuing the
example from above: `#1 [commit message]`. 
4. When finished writing code (see the code checklist below),
submit a pull request and request a review. 
5. After we discuss and review, then *rebase* the branch into main and delete
the branch.

## Code Checklist
When working on large scale engineering projects, one of the most important things
is code quality. Without good code quality collaboration becomes tedious and
difficult. Even though we are a small team, we will adhere to strict styles and
proper testing. This will make it so that our code is not only readable to each other
but also useful to the broader scientific community. 

Before submitting a pull request, here is a checklist of things that our code
should have

- [ ] All (public) functions should have Google docstrings 
(https://google.github.io/styleguide/pyguide.html) with properly labeled input
and return types. 
- [ ] Code should be formatted and styled properly using `black`. Before commiting
anything make sure to run `black .` at the highest level in your directory.
- [ ] Unit tests. If we are adding new code, does our code have appropriate unit tests
and after we make our changes, do any of the old unit tests break?
- [ ] Reference the correct issue when making your commit and pull request. This can be
done by simply including `#n` in your commit message where `n` is replaced with the
corresponding issue number. 
- [ ] Refernce the correct issue when making your pull requests. As with the commits, make
sure to also include `#n` in the description of your pull request message.

## Paper Checklist
- [ ] Do the new `tex` changes compile?
- [ ] Keep lines to be less than 89 characters long and no indentation anywhere in the file.
- [ ] Same as the code changes, reference the correct issue when making commits and pull
requests in the exact same way. 

## Unit Testing
Unit testing will ensure that every function we write operates correctly. Even though
progress will be slower, this will also making scaling our code in the longer run
easier since new features will not break old ones. 

We will do all unit testing through pytest. First, make sure that `pytest` is installed.
```bash
pip install pytest
```

To run unit tests, simply run the following in the terminal
```
pytest .
```
make sure to execute this in the root `mech-arith/` directory. 

### Writing Unit Tests
The `pytest` documentation can be found here (https://docs.pytest.org/en/stable/).
