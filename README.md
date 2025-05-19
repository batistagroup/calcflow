# CalcFlow

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/batistagroup/calcflow/blob/main/LICENSE)
[![image](https://img.shields.io/pypi/v/calcflow.svg)](https://pypi.org/project/calcflow/)
[![codecov](https://codecov.io/gh/batistagroup/calcflow/graph/badge.svg?token=bO5X75J8li)](https://codecov.io/gh/batistagroup/calcflow)
![codequal](https://github.com/batistagroup/calcflow/actions/workflows/quality.yml/badge.svg)

A Python package to simplify preparing and parsing quantum chemistry calculations, with no external dependencies (is true now, will be true forever). Provides a clean, Pythonic API for interacting with QChem, ORCA, and potentially other quantum chemistry software.

## Features

- **No Dependencies**: Built entirely on Python standard library
- **Immutable Objects**: Dataclass-based with a fluent API for transparent, thread-safe state changes
- **Strongly Typed**: Comprehensive type hints and validation for safer code
- **Comprehensive Validation**: Prevent errors before they happen with rigorous validation of inputs
- **Program-Agnostic API**: Abstract away program-specific syntax with a consistent interface
- **Extensible Design**: Abstract base classes make it easy to add support for additional programs

