def getVersion():
    try:
        from importlib.metadata import version
        return version("drbutil")
    except BaseException:
        pass

    try:
        import os
        import json
        pyproject = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(__file__))), "../../", "pyproject.toml"))
        with open(pyproject) as f:
            return next(json.loads(L.split("=")[1]) for L in f if "version" in L)
    except BaseException:
        pass

    return None
    
__version__ = getVersion()

if __name__ == '__main__':
    print(__version__)