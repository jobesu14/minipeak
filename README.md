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
* [mypy](https://github.com/python/mypy) is an optional static type checker for Python.
  See the config in `.mypy.ini`.

## Manually

To test manually the code quality, go at the root of `minipeak` project:

* Unit testing: `pytest`
* Linter: `flake8`
* Static type checker: `mypy .`

# Todo

- [x] implement training summary (hyper-parameters, network parameters, accuracy, precision, recall, loss, ...)
- [x] implement peak position model output
- [x] implement inference code example on a new datasets
- [x] use a logger
- [ ] code documentation
- [ ] mypy and flake formatting (no need of HTML doc)
- [ ] improve readme so that people understand wht this repo is about
- [x] refactore
- [ ] resample abf timeserie based on Hz that user can set as param instead of hard coded sampling
(preprocessing read_abf() and remove_low_freq_trend())
- [ ] write unit test for core functions
- [ ] (implement manual peak detection when peak prediction is true (find max amplitude in window?, use peak derivative?, other?))
