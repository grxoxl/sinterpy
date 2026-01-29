# sinterpy

A small toolkit for seismic inversion and numerical optimization.

Author: Korchuganov V.D.

## Features

- Loss functions for residual vectors: `L2Loss`, `L1Loss` (smoothed), `HuberLoss`
- Model-based objective with optional prior regularization: `ModelBasedObjective`
- Operators: `OperatorBase`, `ConvolutionOperator`, `AcousticStationaryOperator`
- DSP helpers: `ricker_wavelet`, `butter_lowpass_filter`

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

## Tests

```bash
python -m unittest -v
```

## Project structure

```
src/sinterpy/
  constants.py
  losses.py
  objective.py
  operators.py
  utils.py
notebooks/
  checkpoint.ipynb
tests/
  test_checkpoint_example.py
```

## Dependencies

- `numpy`
- `scipy`
- `matplotlib` (used by `notebooks/checkpoint.ipynb`)

## License

Copyright (c) 2026 Korchuganov V.D. All rights reserved.

This code is proprietary. No permission is granted to use, copy, modify, or distribute it without explicit written consent from the author.
