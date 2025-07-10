# Dev notes

This doc is for devs who are working on adding new examples in various frameworks to this repo.

## Pre Commit Hooks

By default there are **NO** pre-commit hooks. But you can check if everything is good to go using:

Run the following before running `git commit`:

```bash 
python -m venv .venv 

# Active the venv (shell dependent)
# ...

# Install pre-commit 
pip install pre-commit

# run pre-commit (will build ALL docker images based on dockerfiles in `experimental` directory of each framework)
pre-commit run
```
