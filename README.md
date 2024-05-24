# drblib

A tiny collection of geometry processing routines frequently used in my prototyping code.

Pure Python, low overhead and minimal dependencies.

## Dependencies

The only **actually required** library is [NumPy](https://github.com/numpy/numpy).

**Optionally**, 
* [matplotlib](https://github.com/matplotlib/matplotlib) shows 2D results,
* [Mayavi](https://github.com/enthought/mayavi) visualizes 3D results and
* [tqdm](https://github.com/tqdm/tqdm) realizes progress bars in the shell.

## Install & Use
Clone the repo and in the main directory, run the `buildAndInstall.bat/.sh` script.

Then you can import everything in your project with `import drblib` or `from drblib import *`, respectively.