import hashlib
import numpy as np

from .util import *

def rgb2hex(rgbCol): return '#'+''.join([hex(int(c))[2:].zfill(2) for c in rgbCol])
def hex2rgb(hexCol): return [int(cv * (1 + (len(hexCol) < 5)), 16) for cv in map(''.join, zip(*[iter(hexCol.replace('#', ''))] * (2 - (len(hexCol) < 5))))]
def seed2hex(seed): return '#' + hashlib.md5(str(seed).encode()).hexdigest()[-6:]
def seed2rgb(seed): return hex2rgb(seed2hex(seed))
def rgb2rgba(rgbCol, alpha=255): return padHor(rgbCol, alpha)

def pad2Dto3D(pts, c=0):
    return np.pad(pts, [[0,0], [0,1]], mode='constant', constant_values=c)

def mat2Dto3D(M):
    return np.block([[M, np.zeros((2,1))],[np.zeros((1,2)), 1]])

def quadsToTris(quadFaces):
    return quadFaces[:,[0,1,2,2,3,0]].reshape(-1,3)

def toEdgeTris(es):
    return np.pad(es, [[0,0],[0,1]], mode='reflect')

def faceToEdges(face):
    face = np.int32(face) if type(face) == list else face
    return face[np.roll(np.repeat(range(len(face)),2), -1)].reshape(-1,2)

def facesToEdges(faces, filterUnique = True):
    es = np.vstack([faceToEdges(face) for face in faces])
    return filterForUniqueEdges(es) if filterUnique else es

def facesToTris(faces):
    if type(faces) == list:
        fLens = list(map(len, faces))
        maxLen = max(fLens)
        mask = np.arange(maxLen) < np.array(fLens)[:,None]
        fcs = np.zeros((len(faces), maxLen), np.int32) - 1
        fcs[mask] = np.concatenate(faces)
        faces = fcs

    tris = np.hstack([np.repeat(faces[:,0].reshape(-1,1), faces.shape[1] - 2, axis=0), np.repeat(faces[:,1:],2, axis=1)[:,1:-1].reshape(-1,2)])
    return tris[np.bitwise_and(tris[:,1]>=0 , tris[:,2]>=0)]

def hexaToEdges(hexa):
    return np.transpose([hexa[[0,1,2,3,4,5,6,7,0,1,2,3]], hexa[[1,2,3,0,5,6,7,4,4,5,6,7]]])

def hexasToEdges(hexas):
    es = np.vstack([hexaToEdges(hexa) for hexa in hexas])
    es.sort(axis=1)
    return unique2d(es)

def tetToEdges(tet):
    return np.transpose([tet[[0,1,2,1,2,3]], tet[[1,2,3,3,0,0]]])

def tetsToEdges(tets):
    es = np.vstack([tetToEdges(tet) for tet in tets])
    es.sort(axis=1)
    return unique2d(es)

tdxs = np.ravel(np.int32([[2,1,0], [3,0,1], [3,1,2], [3,2,0]]))
def tetraToFaces(tetra):
    return tetra[tdxs].reshape(-1,3)
    #return np.vstack([np.roll(tetra, i)[:-1] for i in range(4)])

def tetrasToFaces(tetras):
    tsdxs = np.tile(tdxs, len(tetras)) + np.repeat(np.arange(len(tetras)), 12) * 4
    return np.ravel(tetras)[tsdxs].reshape(-1,3)
    #return np.hstack([np.roll(tetras, i, axis=1)[:,:-1] for i in range(4)]).reshape(-1,3)

def hexaToTetras(hexa):
    return np.vstack([hexa[idxs] for idxs in [[0,1,3,5],[1,2,3,5],[0,3,4,5],[2,3,5,6],[3,4,5,7],[3,5,6,7]]])

def hexaToFaces(hexa):
    return hexa[sixCubeFaces]

def hexasToFaces(hexas):
    return np.vstack(hexas[...,sixCubeFaces])

def edgesToPath(edgesIn):
    edges = copy.deepcopy(edgesIn) if type(edgesIn) == list else edgesIn.tolist()
    nLim = np.arange(len(edges) - 1).sum()
    nTries = 0
    face = edges.pop(0)
    while len(edges):
        edge = edges.pop(0)
        if face[0] == edge[0]:
            face.insert(0, edge[1])
        elif face[-1] == edge[0]:
            face.append(edge[1])
        elif face[0] == edge[1]:
            face.insert(0, edge[0])
        elif face[-1] == edge[1]:
            face.append(edge[0])
        else:
            edges.append(edge)
            nTries += 1
        if nTries > nLim:
            return
    return face if face[0] != face[-1] else face[:-1]

def edgesToPaths(edgesIn):
    edges = copy.deepcopy(edgesIn) if type(edgesIn) == list else edgesIn.tolist()
    face = edges.pop(0)
    faces = []
    iters = 0
    while len(edges):
        edge = edges.pop(0)
        if face[0] == edge[0]:
            face.insert(0, edge[1])
            iters = 0
        elif face[-1] == edge[0]:
            face.append(edge[1])
            iters = 0
        elif face[0] == edge[1]:
            face.insert(0, edge[0])
            iters = 0
        elif face[-1] == edge[1]:
            face.append(edge[0])
            iters = 0
        else:
            edges.append(edge)
            iters += 1
        if len(edges) and iters >= len(edges):
            iters = 0
            faces.append(copy.deepcopy(face if face[0] != face[-1] else face[:-1]))
            face = edges.pop(0)
    faces.append(face if face[0] != face[-1] else face[:-1])
    return faces
