py -m build
python src\drblib\__version__.py > tmp
SET /p drbVersion=<tmp
pip install .\dist\drblib-%drbVersion%-py3-none-any.whl --force-reinstall
DEL tmp