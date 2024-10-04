import os
import sys
import multiprocessing as mp
import logging
import numpy as np
import builtins

from timeit import default_timer as time

np.random.seed(23)

# useful global constants
IN_IDLE = 'idlelib.run' in sys.modules
cpuCount = int(sys.argv[-1]) if '-t' in sys.argv else mp.cpu_count()
builtins.eps = 0.000001
builtins.dotEps = np.deg2rad(0.1)
builtins.quadVerts = np.float32([[-1,-1],[-1,1],[1,1],[1,-1]])
builtins.cubeVerts = np.float32([[-1,-1,-1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,1],[-1,1,1],[1,1,1],[1,-1,1]])
builtins.sixCubeFaces = np.int32([[0,1,2,3],[4,0,3,7],[5,1,0,4],[7,3,2,6],[1,5,6,2],[5,4,7,6]])

# log and result directories
logDir = 'logs/'
resDir = 'results/'
for pDir in [logDir, resDir]:
    if not os.path.exists(pDir):
        try:
            os.mkdir(pDir)
        except PermissionError:
            pass

# for visualizing 2D results
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons
except ImportError:
    mplFound = False
else:
    def simplePlot(xs, ys = None, labels = None):
        for i, x in enumerate(xs):
            plt.plot(np.arange(len(x)) if ys is None else ys[i], x)
        if labels is not None:
            plt.legend(labels)
        plt.show()

    mplFound = True


# for visualizing 3D results
try:
    from mayavi import mlab
    from traits.api import HasTraits, Range, Instance, on_trait_change, Button, Enum
    from traitsui.api import View, Item, Group, HSplit
    from mayavi.core.api import PipelineBase
    from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
except ImportError:
    mlabFound = False
else:
    mlabFound = True


# show progress in shell
try:
    if IN_IDLE:
        raise RuntimeError
    from tqdm import tqdm
except:

    class tqdmDummy:
        n = 0

        def update(self, x):
            return

        def close(self):
            return

    def tqdm(x, **kwargs):
        return tqdmDummy() if x is None else x


# simple logger class
logging.basicConfig(format='%(message)s')
class Logger:
    def __init__(self, logName):
        self.log = logging.getLogger(logName)
        self.log.setLevel(logging.INFO)
        self.log.addHandler(logging.FileHandler(logDir + '%s.log' % logName, mode='w'))

    def logThis(self, msg, args, style=None):
        tplArgs = args if not hasattr(args, '__len__') or type(args) == str else tuple(args)
        fmtArgs = style % tplArgs if style is not None else str(tplArgs)
        self.log.info(msg + ':\t' + fmtArgs)

# simple memoizer, use with @memoize decorator
class memoize:
    def __init__(self, f):
        self.f = f
        self.memDict = {}
    def __call__(self, *args):
        if args not in self.memDict:
            self.memDict[args] = self.f(*args)
        return self.memDict[args]
builtins.memoize = memoize

from .__version__ import __version__
from .util import *
from .io import *
