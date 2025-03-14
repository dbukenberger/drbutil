# drbutil
[![PyPI version](https://badge.fury.io/py/drbutil.svg)](https://badge.fury.io/py/drbutil)

A tiny collection of geometry processing routines frequently used in my prototyping code.

Pure Python, low overhead and minimal dependencies.

## Dependencies

The only **actually required** library is [NumPy](https://github.com/numpy/numpy).

**Optionally**, 
* [matplotlib](https://github.com/matplotlib/matplotlib) shows 2D results,
* [Mayavi](https://github.com/enthought/mayavi) visualizes 3D results,
* [tqdm](https://github.com/tqdm/tqdm) realizes progress bars in the shell and
* [scipy](https://github.com/scipy/scipy) speeds up sparse solving operations.

## Install & Use
Install from [PyPI](https://pypi.org/project/drbutil/) with `pip install drbutil` or
clone the repo and run the `buildAndInstall.bat/.sh` script.

Then you can import everything in your project with `import drbutil` or `from drbutil import *`, respectively.
