#! /bin/bash
py -m build
drbVersion=`python ./src/drblib/__version__.py`
pip install ./dist/drblib-$drbVersion-py3-none-any.whl --force-reinstall