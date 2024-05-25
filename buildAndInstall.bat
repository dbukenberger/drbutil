py -m build
python src\drbutil\__version__.py > tmp
SET /p drbVersion=<tmp
pip install .\dist\drbutil-%drbVersion%-py3-none-any.whl --force-reinstall
DEL tmp