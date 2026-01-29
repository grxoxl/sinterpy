# sinterpy

A small toolkit for seismic inversion and numerical optimization.

Author: Korchuganov V.D.

## Installation

Requirements: Python `>=3.10`.

Option 1 (recommended; uses `pyproject.toml`):

```bash
python -m pip install -e .
```

Option 2 (via `requirements.txt`):

```bash
python -m pip install -r requirements.txt
```

## Project structure

```
src/sinterpy/
  constants.py
  custom_loss.py
  custom_objective.py
  custom_operator.py
  custom_utils.py
notebooks/
  checkpoint.ipynb
```

## Dependencies

- `numpy`
- `scipy`
- `matplotlib` (used by `notebooks/checkpoint.ipynb`)

## License

Not specified (`UNLICENSED` in `pyproject.toml`).
