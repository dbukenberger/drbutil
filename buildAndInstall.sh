#! /bin/bash
py -m build
drbVersion=`python ./src/drbutil/__version__.py`
pip install ./dist/drbutil-$drbVersion-py3-none-any.whl --force-reinstall