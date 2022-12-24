# Minipeak #
A Python package to automatically detect minis peaks from electrophysiology.
The electrophysiology data are expected to be in a abf file format.

# Installation

## Requirements

### User

Installing Gastonpy locally:
```
pip3 install .
```

Then, from anywhere on your system, you can use the scripts declared in
`setup.py` in `entry_points`. For example:
```
gastonpy_dummy
```

### Developper

Optionally, you can use the `pip3 install -e .` option to install in editable mode.

To install the tools for developpement (listed in `setup.py` in `extras_require`), use:
```
pip3 install .[dev]
```

# Code quality

To improve Python code quality, the following frameworks have been choosen:

* [pytest](https://docs.pytest.org/en/6.2.x/) automated tests framework that supports unittest
* [flake8](https://flake8.pycqa.org/en/latest/index.html) (See the config in `.flake8`)
  is a wrapper around these tools:
    * PyFlakes: a simple program which checks Python source files for errors.
    * pycodestyle: is a tool to check your Python code against some of the style
      conventions in PEP 8.
    * Ned Batchelder's McCabe script: check McCabe complexity.
* [mypy](https://github.com/python/mypy) is an optional static type checker for Python.
  See the config in `.mypy.ini`.
    * [PEP 484 -- Type Hint](https://www.python.org/dev/peps/pep-0484/): a Python
      Enhancement Proposal providing a standard syntax for type annotations.
    * [Type hints cheat sheet (Python 2)](https://mypy.readthedocs.io/en/stable/cheat_sheet.html):
      shows how the PEP 484 type language represents various common types in Python 2.
    * [Type hints cheat sheet (Python 3)](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html):
      shows how the PEP 484 type annotation notation represents various common types
      in Python 3.

## Manually

To test manually the code quality, go at the root of `gastonpy` project:

* Unit testing: `pytest`
* Linter: `flake8`
* Static type checker: `mypy .`
