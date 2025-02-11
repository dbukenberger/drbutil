import hashlib
import copy
import numpy as np


# converting helpers

def rgb2hex(rgbCol): return '#'+''.join([hex(int(c))[2:].zfill(2) for c in rgbCol])
def hex2rgb(hexCol): return [int(cv * (1 + (len(hexCol) < 5)), 16) for cv in map(''.join, zip(*[iter(hexCol.replace('#', ''))] * (2 - (len(hexCol) < 5))))]
def seed2hex(seed): return '#' + hashlib.md5(str(seed).encode()).hexdigest()[-6:]
def seed2rgb(seed): return hex2rgb(seed2hex(seed))
def rgb2rgba(rgbCol, alpha=255): return padHor(rgbCol, alpha)

def pad2Dto3D(pts, c=0):
    return np.pad(pts, [0,1] if pts.ndim == 1 else [[0,0], [0,1]], mode='constant', constant_values=c)

def mat2Dto3D(M):
    return np.block([[M, np.zeros((2,1))],[np.zeros((1,2)), 1]])

def quadsToTris(quadFaces):
    return quadFaces[:,[0,1,2,2,3,0]].reshape(-1,3)

def toEdgeTris(es):
    return np.pad(es, [[0,0],[0,1]], mode='reflect')

def faceToEdges(face):
    return np.transpose([face, np.roll(face, -1)])

def facesToEdges(faces, filterUnique = True):
    if type(faces) == list:
        es = np.vstack([faceToEdges(face) for face in faces])
    else:
        es = np.transpose([faces.ravel(), np.roll(faces, -1, axis=1).ravel()])
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

def hexOrderBT2Dg(hexa):
    o = [0,4,3,1,7,5,2,6]
    return hexa[o] if hexa.ndim == 1 else hexa[:,o]

def hexOrderDg2BT(hexa):
    o = [0,2,6,3,1,4,7,5]
    return hexa[o] if hexa.ndim == 1 else hexa[:,o]

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

def edgesToPath(edgesIn, withClosedFlag = False, returnPartial = False):
    edges = copy.deepcopy(edgesIn) if type(edgesIn) == list else edgesIn.tolist()
    path = edges.pop(0)
    iters = 0
    while len(edges):
        edge = edges.pop(0)
        if path[0] == edge[0]:
            path.insert(0, edge[1])
            iters = 0
        elif path[-1] == edge[0]:
            path.append(edge[1])
            iters = 0
        elif path[0] == edge[1]:
            path.insert(0, edge[0])
            iters = 0
        elif path[-1] == edge[1]:
            path.append(edge[0])
            iters = 0
        else:
            edges.append(edge)
            iters += 1
        if len(edges) and iters > len(edges):
            if returnPartial:
                break
            else:
                return
    if path[0] == path[-1]:
        return (path[:-1], True) if withClosedFlag else path[:-1]
    else:
        return (path, False) if withClosedFlag else path

def edgesToPaths(edges, withClosedFlag = False):
    vIdxs = np.unique(flatten(edges) if type(edges) == list else edges.ravel())
    adj = {vIdx: [] for vIdx in vIdxs}
    for vIdx, vJdx in edges:
        adj[vIdx].append(vJdx)
        adj[vJdx].append(vIdx)

    if max([len(adj[v]) for v in vIdxs]) > 2:
        warnings.warn('Only vetex valcences <= 2 are supported.')
        return

    visited = set()
    paths, closed = [], []
    for vIdx in vIdxs:
        if vIdx in visited:
            continue

        nIdxs = adj[vIdx]
        path = [nIdxs[0], vIdx, nIdxs[1]] if len(nIdxs) == 2 else [nIdxs[0], vIdx]
        isClosed = False
        endsReached = [True, True]
        while any(endsReached) and not isClosed:
            for idxs in [[0,1,-1], [-1,-2,0]]:
                if endsReached[idxs[0]]:
                    nIdxs = adj[path[idxs[0]]]
                    if len(nIdxs) == 1 and nIdxs[0] == path[idxs[1]]:
                        endsReached[idxs[0]] = False
                    else:
                        vJdx = nIdxs[0] if nIdxs[0] != path[idxs[1]] else nIdxs[1]
                        if vJdx == path[idxs[2]]:
                            isClosed = True
                            break
                        path = path + [vJdx] if idxs[0] < 0 else [vJdx] + path

        paths.append(path)
        closed.append(isClosed)
        visited.update(path)
        
    return (paths, closed) if withClosedFlags else paths

def pathToEdges(path, closed = True):
    es = np.transpose([path, np.roll(path, -1)])
    return es if closed else es[:-1]

def groupEdges(edges):
    return [pathToEdges(path, closed) for path, closed in zip(*edgesToPaths(edges, True))]

def quaternionToMatrix(q):
    w, x, y, z = q
    return np.float32([[1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                       [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                       [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])

def matrixToQuaternion(m):
    # Paper: New Method for Extracting the Quaternion from a Rotation Matrix
    [d11,d12,d13],[d21,d22,d23],[d31,d32,d33] = m
    """
    K2 = np.float32([[d11-d22, d21+d12, d31, -d32],
                     [d21+d12, d22-d11, d32, d31],
                     [d31, d32, -d11-d22, d12-d21],
                     [-d32, d31, d12-d21, d11+d22]]) / 2
    eVals, eVecs = np.linalg.eig(K2)
    """

    K3 = np.float32([[d11-d22-d33, d21+d12, d31+d13, d23-d32],
                     [d21+d12, d22-d11-d33, d32+d23, d31-d13],
                     [d31+d13, d32+d23, d33-d11-d22, d12-d21],
                     [d23-d32, d31-d13, d12-d21, d11+d22+d33]]) / 3
    eVals, eVecs = np.linalg.eig(K3)
    #return eVecs[np.argmax(np.abs(eVals))]

    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/ code by angel

    [m00,m01,m02],[m10,m11,m12],[m20,m21,m22] = m    
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m21 - m12) * s
        qy = (m02 - m20) * s
        qz = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s
            
    return np.float32([qw,qx,qy,qz])


# basic geometry and utility functions

def inner1d(u, v): return (u * v).sum(axis=u.ndim-1)

def normVec(v, withLen = False):
    if v.ndim == 1:
        n = np.sqrt(np.dot(v, v))
        res = v / n if n else v * 0
        return (res, n) if withLen else res
    else:
        n = np.sqrt(inner1d(v, v))
        m = n != 0
        v = v.copy()
        v[m] /= n[m].reshape(-1, 1)
        return (v, n) if withLen else v

def norm(v): return np.sqrt(np.dot(v, v) if v.ndim == 1 else inner1d(v, v))

def normZeroToOne(data): return (np.float32(data) - np.min(data))/(np.max(data) - np.min(data)) if len(data) else data

def mnmx(v): return np.float32([v.min(axis = 0), v.max(axis=0)])

def unique2d(a): return a[np.unique(a[:,0] + a[:,1]*1.0j,return_index=True)[1]]

def orthoVec(v): return [1, -1] * (v[::-1] if v.ndim == 1 else v[:, ::-1])

def randomJitter(n, k, s=1): return normVec(np.random.rand(n, k) * 2 - 1) * np.random.rand(n, 1) * s

def generateJitteredGridPoints(n, d, e=1): return generateGridPoints(n, d, e) + randomJitter(n**d, d, e / n)

def vecsParallel(u, v, signed=False): return 1 - np.dot(u, v) < eps if signed else 1 - np.abs(np.dot(u, v)) < eps

def distPointToPlane(p, o, n): return np.abs(np.dot(p - o, n))

def planesEquiv(onA, onB): return distPointToPlane(onA[0], onB[0], onB[1]) < eps and vecsParallel(onA[1], onB[1])

def inner3x3M(As, Bs): return inner1d(np.repeat(As.reshape(-1,3),3,axis=0), np.repeat(np.transpose(Bs,(0,2,1)).reshape(-1,9),3,axis=0).reshape(-1,3)).reshape(-1,3,3)

def innerVxM(vs, Ms): return np.einsum('ij,ikj->ik', vs, Ms) # np.vstack([np.dot(v, M.T) for v, M in zip(vs, Ms)])

def innerNxM(Ns, Ms): return np.einsum('ijh,ikh->ijk', Ns, Ms) # np.vstack([np.dot(N, M.T) for N, M in zip(Ns, Ms)])

def innerAxBs(A, Bs): return np.einsum('jk,ijk->ij', A, Bs) # np.vstack([inner1d(A, B) for B in Bs])

def outer1d(us, vs): return np.einsum('ij,ih->ijh', us, vs)

def outer2dWeighted(us, vs, ws): return np.sum([outer1d(us[:,d], vs[:,d]) * ws[:,d].reshape(-1,1,1) for d in range(us.shape[1])], axis=0)
#def outer2dWeighted(us, vs, ws): return np.einsum('ijk,ijh,ij->ikh', us, vs, ws) # equivalent but slower =(

def dotAxBs(A, Bs): return np.einsum('jk,ijl->ikl', A, Bs) # np.float32([np.dot(A, B) for B in Bs])

def Mr2D(a): return np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def Mr(ori): return Mr2D(ori) if np.isscalar(ori) else Mr3D(ori[0], ori[1], ori[2])

def padHor(pts, c=0): return np.pad(pts, [[0,0],[0,1]], mode='constant', constant_values=c)

def flatten(lists): return [element for elements in lists for element in elements]

def cantorPi(k1, k2): return ((k1 + k2) * (k1 + k2 + 1)) // 2 + (k1 if k1 > k2 else k2)
def cantorPiO(k1, k2): return ((k1 + k2) * (k1 + k2 + 1)) // 2 + k2
def cantorPiK(kxs): return cantorPi(kxs[0], kxs[1]) if len(kxs) < 3 else cantorPi(cantorPiK(kxs[:-1]), kxs[-1])


def cantorPiV(k1k2, sort = True):
    if k1k2.dtype != np.int64:
        k1k2 = np.int64(k1k2)
    if sort:
        k1k2 = k1k2.copy()
        k1k2.sort(axis=1)
    k1k2sum = k1k2[:, 0] + k1k2[:, 1]
    return ((k1k2sum * k1k2sum + k1k2sum) >> 1) + k1k2[:, 1]

def cantorPiKV(kKs, sort = True):
    if sort:
        kKs = kKs.copy()
        kKs.sort(axis=1)
    return cantorPiV(np.transpose([cantorPiKV(kKs[:,:-1], False), kKs[:,-1]]) if kKs.shape[1] > 2 else kKs)

def simpleSign(xs, thresh=None):
    signs = np.int32(np.sign(xs))
    if thresh is None:
        return signs
    else:
        return signs * (np.abs(xs) > thresh)

def simpleDet(M):
    return simpleDet2x2(M) if len(M) == 2 else simpleDet3x3(M)

def simpleDets(Ms):
    return simpleDets2x2(Ms) if len(Ms[0]) == 2 else simpleDets3x3(Ms)

def simpleDet2x2(M):
    a,b,c,d = M[[0,0,1,1],[0,1,0,1]]
    return a*d-b*c

def simpleDets2x2(M):
    a,b,c,d = M[:,[0,0,1,1],[0,1,0,1]].T
    return a*d-b*c

def simpleDet3x3(M):
    a,b,c,d,e,f,g,h,i = M[[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]]
    return a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

def simpleDets3x3(Ms):
    a,b,c,d,e,f,g,h,i = Ms[:,[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]].T
    return a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

def simpleSVD(A, returnS = True):
    evals, V = np.linalg.eigh(np.dot(A.T, A))
    V = V[:, ::-1]
    svals = np.sqrt(np.maximum(evals[::-1], 0))

    m = svals > eps
    U = np.dot(A, V)
    U *= m
    U[:,m] /= svals[m]

    return (U, svals, V.T) if returnS else (U, V.T)

def simpleSVDs(As, returnS = True):
    evalss, Vs = np.linalg.eigh(np.transpose(As, axes=[0,2,1]) @ As)
    Vs = Vs[:,:,::-1]
    svalss = np.sqrt(np.maximum(evalss[:,::-1], 0))

    nDim = evalss.shape[1]
    ms = svalss > eps
    Us = As @ Vs
    Us *= ms.reshape(-1, 1, nDim)
    mms = np.repeat(ms[:,np.newaxis], nDim, axis=1)
    Us[mms] /= np.repeat(svalss[:,np.newaxis], nDim, axis=1)[mms]

    Vts = np.transpose(Vs, axes=[0,2,1])
    return (Us, svalss, Vts) if returnS else (Us, Vts)

def cross(a, b, normed=False):
    if a.ndim == 1 and b.ndim == 1:
        c = np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
    if a.ndim == 1 and b.ndim > 1:
        c = np.array([a[1]*b[:,2]-a[2]*b[:,1],a[2]*b[:,0]-a[0]*b[:,2],a[0]*b[:,1]-a[1]*b[:,0]]).T
        #c = np.array([b[:,1]*a[2]-b[:,2]*a[1],b[:,2]*a[0]-b[:,0]*a[2],b[:,0]*a[1]-b[:,1]*a[0]]).T
    if a.ndim > 1 and b.ndim == 1:
        return cross(b, a, normed)
    if a.ndim > 1 and b.ndim > 1:
        c = np.array([a[:,1]*b[:,2]-a[:,2]*b[:,1],a[:,2]*b[:,0]-a[:,0]*b[:,2],a[:,0]*b[:,1]-a[:,1]*b[:,0]]).T
    return normVec(c) if normed else c

def Mr3D(alpha=0, beta=0, gamma=0):
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))

def icdf(xs): # inverse cumulative distribution function
    uxs, iIdxs, cts = np.unique(xs, return_inverse = True, return_counts = True)
    ccts = np.cumsum(cts)-cts[0]
    return np.clip(ccts[iIdxs] / (len(xs) - cts[0] - cts[-1]), 0, 1)

def generateGridPoints(n, d, e=1):
    ptsGrid = np.linspace(-e, e, n, endpoint=False) + e / n
    if d == 1:
        return ptsGrid
    if d == 2:
        return np.vstack(np.dstack(np.meshgrid(ptsGrid, ptsGrid)))
    elif d == 3:
        return np.float32(np.vstack(np.vstack(np.transpose(np.meshgrid(ptsGrid, ptsGrid, ptsGrid), axes=[3,2,1,0]))))
    else:
        warnings.warn('%d dimensions not supported'%d)
        return

def generatePointsOnCircle(n, in3D = False, withCenter = False):
    if n<=0:
        return None
    offset = np.pi/n
    alpha = np.linspace(offset, np.pi*2-offset, n)
    pts = np.transpose([np.cos(alpha), np.sin(alpha)])
    if withCenter:
        pts = np.vstack([pts, [0,0]])
    return pad2Dto3D(pts) if in3D else pts

def generateIcoSphere(nSubdiv = 0, likeBlender = False):
    if nSubdiv < 0:
        verts = generatePointsOnCircle(3, True, True)
        verts[:,-1] += np.float32([-1,-1,-1,3]) * np.sqrt(2)/4
        tris = np.int32([[0,2,1],[0,1,3],[1,2,3],[0,3,2]])
        return normVec(verts), tris
    
    t = (1 + np.sqrt(5)) / 2
    if likeBlender:
        # blender subdiv is different, but okay for first two levels
        verts = normVec(np.float32([[-t,0,-1],[0,-1,-t],[-1,-t,0],[-t,0,1],[-1,t,0],[0,1,-t],[1,-t,0],[0,-1,t],[0,1,t],[1,t,0],[t,0,-1],[t,0,1]]))
        verts = np.dot(verts, Mr3D(0, np.arctan2(verts[-1,0],verts[-1,2]), 0).T)
        tris = np.int32([[0,1,2],[1,0,5],[0,2,3],[0,3,4],[0,4,5],[1,5,10],[2,1,6],[3,2,7],[4,3,8],[5,4,9],[1,10,6],[2,6,7],[3,7,8],[4,8,9],[5,9,10],[6,10,11],[7,6,11],[8,7,11],[9,8,11],[10,9,11]])
    else:
        verts = normVec(np.float32([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]]))
        tris = np.int32([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]])

    for i in range(nSubdiv):
        edges = facesToEdges(tris)
        eHashVertIdxs = {eHash: p for eHash, p in zip(cantorPiV(edges), len(verts)+np.arange(len(edges)))}
        newTris = []
        for tri in tris:
            u0, u1, u2 = [eHashVertIdxs[eHash] for eHash in cantorPiV(faceToEdges(tri))]
            newTris += [[tri[0],u0,u2],[tri[1],u1,u0],[tri[2],u2,u1],[u0,u1,u2]]
        tris = np.int32(newTris)
        verts = np.vstack([verts, normVec(verts[edges].mean(axis=1))])

    return verts, tris

def generateSamples(vs, elems, n = None, weights = None):
    nDim = elems.shape[1]
    if n is None:
        n = len(elems)
    if weights is None:
        weights = norm(vs[elems[:,0]] - vs[elems[:,1]]) if nDim == 2 else computeTriangleAreas(vs[elems], False)

    wCum = np.cumsum(weights)
    randomIdxs = np.searchsorted(wCum, np.random.rand(n) * wCum[-1])
    # faster but idxs array can get huuge
    #idxs = np.repeat(np.arange(len(elems)), np.int32(weights / weights.min()))
    #randomeIdxs = idxs[np.random.randint(0, len(idxs), n)]
           
    bWeights = np.random.rand(n, nDim)**(nDim-1)
    bWeights /= bWeights.sum(axis=1).reshape(-1,1)
    return (bWeights[:,:,np.newaxis] * vs[elems[randomIdxs]]).sum(axis=1)

@memoize
def generateTriGridSampleWeights(n, innerOnly = False):
    n = max(2,n+2)
    ws = []
    for i in range(innerOnly, n):
        for j in range(innerOnly, n - i):
            k = (n-1) - i - j
            if innerOnly and not k:
                continue
            ws.append([i,j,k])
    return np.float32(ws) / (n-1)

@memoize
def generateQuadGridSampleWeights(n, innerOnly = False):
    n = max(2,n+2)
    ws = np.linspace(0,1,n)
    if innerOnly:
        ws = ws[1:-1]
    u,v = np.dstack(np.meshgrid(ws, ws)).reshape(-1,2).T
    return np.transpose([(1-u)*(1-v), u*(1-v), u*v, (1-u)*v])

@memoize
def generateTetGridSampleWeights(n, innerOnly = False):
    n = max(2,n+2)
    ws = []
    for i in range(innerOnly, n):
        for j in range(innerOnly, n - i):
            for k in range(innerOnly, n - i - j):
                l = (n-1) - i - j - k
                if innerOnly and not l:
                    continue
                ws.append([i,j,k,l])
    return np.float32(ws) / (n-1)

@memoize
def generateHexGridSampleWeights(n, innerOnly = False):
    n = max(2,n+2)
    ws = np.linspace(0,1,n)
    if innerOnly:
        ws = ws[1:-1]
    u,v,w = np.stack(np.meshgrid(ws, ws, ws), axis=3).reshape(-1, 3).T
    return np.transpose([(1-u)*(1-v)*(1-w), u*(1-v)*(1-w), u*v*(1-w), (1-u)*v*(1-w), (1-u)*(1-v)*w, u*(1-v)*w, u*v*w, (1-u)*v*w])

def generateTriGridSamples(pts, n, innerOnly = False):
    return np.dot(generateTriGridSampleWeights(n, innerOnly), pts) if n-2*innerOnly >= 0 else np.empty((0, pts.shape[1]), np.float32)

def generateQuadGridSamples(pts, n, innerOnly = False):
    return np.dot(generateQuadGridSampleWeights(n, innerOnly), pts)

def generateTetGridSamples(pts, n, innerOnly = False):
    return np.dot(generateTetGridSampleWeights(n, innerOnly), pts) if n-3*innerOnly >= 0 else np.empty((0, pts.shape[1]), np.float32)

def generateHexGridSamples(pts, n, innerOnly = False):
    return np.dot(generateHexGridSampleWeights(n, innerOnly), pts)

@memoize
def generateTriGridTriangles(n, innerOnly = False):
    if innerOnly and n <= 2:
        return np.empty([0,3], np.int32)
    tris = []
    n += 1 - 3*innerOnly
    s, o = 0, n+1
    for r in range(n):
        for t in range(s, s+n-r):
            tris.append([t+r, t+r+1, t+r+o])
            if r:
                tris.append([t+r, t+r-o, t+r+1])
        o -= 1
        s += o
    return np.int32(tris)

@memoize
def generateQuadGridQuads(n, innerOnly = False):
    if innerOnly and n <= 1:
        return np.empty([0,4], np.int32)
    n += 1 - 2*innerOnly
    m = n+1
    vIdxs = lambda i: [i, i+1, i+1+m, i+m]
    ijs = np.vstack([[i,j] for j in range(n) for i in range(n)])
    return np.vstack([vIdxs(i+m*j) for i,j in ijs])

@memoize
def generateTetGridTetrahedra(n, innerOnly = False): # not a fully regular fractal subdivision
    if innerOnly and n <= 3:
        return np.empty([0,4], np.int32)
    tets = []
    n += 1 - 4*innerOnly
    offs = np.cumsum(np.arange(n+2))[::-1]
    s, o = 0, n+1
    for z in range(n):
        off = int(np.sum(offs[:z]))
        for y in range(n-z):
            for x in range(n-z-y):
                shift = off + s + x
                tets.append(np.int32([0, 1, o, offs[z]-y]) + shift)
                if x: # weird inner octahedron as 4 tets
                    a,b,c,d,e,f = np.int32([0, o-1, o, offs[z]-y-1, offs[z]-y, offs[z]-y-1+o-1]) + shift
                    tets += [[a,c,b,f], [a,b,d,f], [a,e,c,f], [a,d,e,f]]
            s += o
            o -= 1
        s, o = 0, n-z
    return np.int32(tets)

@memoize
def generateHexGridHexahedra(n, innerOnly = False):
    if innerOnly and n <= 1:
        return np.empty([0,8], np.int32)
    n += 1 - 2*innerOnly
    m = n+1
    vIdxs = lambda i: [i+1, i+m+1, i+m+m*m+1, i+1+m*m, i, i+m, i+m+m*m, i+m*m]
    ijks = np.vstack([[i,j,k] for k in range(n) for j in range(n) for i in range(n)])
    return np.vstack([vIdxs(i+m*j+m*m*k) for i,j,k in ijks])

def generateTriGrid(n, in3D = False, innerOnly = False):
    vs = generateTriGridSamples(generatePointsOnCircle(3, in3D), n, innerOnly)
    ts = generateTriGridTriangles(n, innerOnly)
    return vs, ts

def generateQuadGrid(n, in3D = False, innerOnly = False):
    vs = generateQuadGridSamples(pad2Dto3D(quadVerts) if in3D else quadVerts, n, innerOnly)
    ts = generateQuadGridQuads(n, innerOnly)
    return vs, ts

def generateTetGrid(n, innerOnly = False):
    vs, ts = generateIcoSphere(-1)
    vs = generateTetGridSamples(vs, n, innerOnly)
    ts = generateTetGridTetrahedra(n, innerOnly)
    return vs, ts

def generateHexGrid(n, innerOnly = False):
    vs = generateHexGridSamples(cubeVerts, n, innerOnly)
    ts = generateHexGridHexahedra(n, innerOnly)
    return vs, ts

def computePointInPolygonKernel(pts, improvementIters = -1):
    n = len(pts)    
    vTris = pad2Dto3D(np.roll(np.repeat(np.arange(n), 2),-1).reshape(-1,2), n)[:,::-1]
    pPts = np.vstack([pts, pts.mean(axis=0)])
    
    ws = generateTriGridSampleWeights(5)
    ws[:,1:] = ws[:,1:]**2
    ws[:,0] = 1-ws[:,1:].sum(axis=1)

    vTriz = np.vstack([vTris + [i,0,0] for i in range(len(ws)*len(vTris))])   
    while True:
        sPts = np.dot(ws, pPts[vTris]).reshape(-1,2)
        pPtz = np.vstack([pts, sPts])
        SJs = computeJacobians(pPtz[vTriz], True).min(axis=1).reshape(len(sPts), -1).min(axis=1)
        cPt = sPts[np.argmax(SJs)]
        if not improvementIters or norm(pPts[-1] - cPt) < eps:
            if max(SJs) < 0:
                warnings.warn('No kernel point found, polygon probably not star-shaped.')
            break
        improvementIters -= 1
        pPts[-1] = cPt
    return cPt

def distPointToEdge(A, B, P, withClosest = False):
    AtoB = B - A
    AtoP = P - A
    BtoP = P - B
    
    if np.dot(AtoB, AtoP) <= 0:
        dist = norm(AtoP)
        closest = A
    elif np.dot(-AtoB, BtoP) <= 0:
        dist = norm(BtoP)
        closest = B
    else:
        d = normVec(AtoB)
        AtoC = np.dot(AtoP, d) * d
        dist = norm(AtoP-AtoC)
        closest = A + AtoC
    return (dist, closest) if withClosest else dist

def distsPointToEdges(ABs, P, withClosest = False):
    As, Bs = ABs[:,0], ABs[:,1]
    AsToBs = Bs - As
    closests = As + AsToBs * np.clip(inner1d(P - As, AsToBs) / inner1d(AsToBs, AsToBs), 0, 1).reshape(-1,1)
    dists = norm(closests - P)
    return (dists, closests) if withClosest else dists

def distPointsToEdge(A, B, Ps):
    AtoB = B - A
    AtoPs = Ps - A
    BtoPs = Ps - B

    mskA = np.dot(AtoPs, AtoB) <= 0
    mskB = np.dot(BtoPs, -AtoB) <= 0
    mskC = np.bitwise_and(mskA^True, mskB^True)

    dists = np.zeros(len(Ps), np.float32)

    dists[mskA] = norm(AtoPs[mskA])
    dists[mskB] = norm(BtoPs[mskB])

    d = normVec(AtoB)
    dists[mskC] = norm(AtoPs[mskC] - np.dot(AtoPs[mskC], d).reshape(-1,1) * d)
    return dists    

def edgesIntersect2D(A, B, C, D):
    if A.ndim == 1:
        d = (D[1] - C[1]) * (B[0] - A[0]) - (D[0] - C[0]) * (B[1] - A[1])
        u = (D[0] - C[0]) * (A[1] - C[1]) - (D[1] - C[1]) * (A[0] - C[0])
        v = (B[0] - A[0]) * (A[1] - C[1]) - (B[1] - A[1]) * (A[0] - C[0])
        if d < 0:
            u, v, d = -u, -v, -d
        return (0 <= u <= d) and (0 <= v <= d)
    d = (D[1] - C[1]) * (B[:,0] - A[:,0]) - (D[0] - C[0]) * (B[:,1] - A[:,1])
    u = (D[0] - C[0]) * (A[:,1] - C[1]) - (D[1] - C[1]) * (A[:,0] - C[0])
    v = (B[:,0] - A[:,0]) * (A[:,1] - C[1]) - (B[:,1] - A[:,1]) * (A[:,0] - C[0])
    dMsk = d < 0
    if dMsk.any():
        u[dMsk] *= -1
        v[dMsk] *= -1
        d[dMsk] *= -1
    return (0 <= u) * (u <= d) * (0 <= v) * (v <= d)

def intersectEdgesWithRay2D(ABs, C, d):
    eDirs = normVec(ABs[:,1] - ABs[:,0])
    #eNormals = np.dot(eDirs, Mr2D(np.pi/2))
    toCs = C - ABs[:,0]

    dNom = d[1] * eDirs[:,0] - d[0] * eDirs[:,1]
    
    ts = toCs[:,0] * d[1] - toCs[:,1] * d[0]
    ss = toCs[:,0] * eDirs[:,1] - toCs[:,1] * eDirs[:,0]

    m = dNom != 0
    ts[m] /= dNom[m]
    ss[m] /= dNom[m]

    m = np.bitwise_or(np.bitwise_or(ts <= 0, ss <= 0), ts > norm(ABs[:,1] - ABs[:,0]))
    ss[m] = ss.max()
    return C + d * ss[ss.argmin()]

def pointInBB(BB, pt):
    return BB[0,0] <= pt[0] and BB[0,1] <= pt[1] and BB[1,0] >= pt[0] and BB[1,1] >= pt[1]

def pointInBB3D(BB, pt):
    return (BB[0,0] <= pt[0]) * (BB[0,1] <= pt[1]) * (BB[0,2] <= pt[2]) * (BB[1,0] >= pt[0]) * (BB[1,1] >= pt[1]) * (BB[1,2] >= pt[2])

def pointsInBB3D(BB, pts):
    return np.bitwise_and(np.all(BB[0] <= pts, axis=1), np.all(BB[1] >= pts, axis=1))

def bbIntersect3D(aBB, bBB):
    return np.all(bBB[1] > aBB[0]) and np.all(bBB[0] < aBB[1])

def bbsIntersect3D(aBB, bBBs):
    return np.bitwise_and(np.all(bBBs[:,1] > aBB[0], axis=1), np.all(bBBs[:,0] < aBB[1], axis=1))

def pointInTriangle2D(A, B, C, P):
    v0, v1, v2 = C - A, B - A, P - A
    u, v, d = v2[1] * v0[0] - v2[0] * v0[1], v1[1] * v2[0] - v1[0] * v2[1], v1[1] * v0[0] - v1[0] * v0[1]
    if d < 0:
        u, v, d = -u, -v, -d
    return u >= 0 and v >= 0 and (u + v) <= d

def pointInTriangles2D(ABCs, P):
    v0s, v1s, v2s = ABCs[:,2] - ABCs[:,0], ABCs[:,1] - ABCs[:,0], P - ABCs[:,0]
    us, vs, ds = v2s[:,1] * v0s[:,0] - v2s[:,0] * v0s[:,1], v1s[:,1] * v2s[:,0] - v1s[:,0] * v2s[:,1], v1s[:,1] * v0s[:,0] - v1s[:,0] * v0s[:,1]
    m = ds < 0
    if m.any():
        us[m] *= -1
        vs[m] *= -1
        ds[m] *= -1
    return np.bitwise_and(np.bitwise_and(us >= 0, vs >= 0), (us + vs) <= ds).any()    

def pointInTriangle3D(A, B, C, P):
    u,v,w = B-A, C-B, A-C
    if np.abs(np.dot(P-A, cross(u, v, True))) > eps:
        return False
    if np.dot(cross(u, -w, True), cross(u, P-A, True)) < 0:
        return False
    if np.dot(cross(v, -u, True), cross(v, P-B, True)) < 0:
        return False
    if np.dot(cross(w, -v, True), cross(w, P-C, True)) < 0:
        return False
    return True

def pointInTriangles3D(ABCs, P, uvws = None, ns = None, assumeInPlane = False):
    if uvws is None:
        uvws = ABCs[:,[1,2,0]] - ABCs
    if assumeInPlane:
        m = np.ones(len(ABCs), np.bool8)
    else:
        m = np.abs(inner1d(P-ABCs[:,0], cross(uvws[:,0], uvws[:,1], True) if ns is None else ns[:,0])) < eps
        if not m.any():
            return False
    for i in range(3):
        m *= inner1d(cross(uvws[:,i], -uvws[:,(i+2)%3], True) if ns is None else ns[:,i], cross(uvws[:,i], P-ABCs[:,i], True)) > 0
        if not m.any():
            return False
    return True

def pointInPolygonKernel2D(pPts, p):
    vTris = pad2Dto3D(np.roll(np.repeat(np.arange(len(pPts)), 2),-1).reshape(-1,2), len(pPts))
    pts = np.vstack([pPts, p])
    areas = computeTriangleAreas(pts[vTris])
    signs = simpleSign(computeTriangleAreas(pts[vTris]), eps)
    return np.all(signs == signs[0])

def pointInPolygon2D(pPts, p):
    #es = pPts[np.roll(np.repeat(np.arange(4), 2), -1).reshape(-1,2)]
    es = np.stack([pPts, np.roll(pPts, -1, axis=0)], axis=1)
    iPts = intersectEdgesEdge2D(es, [p, pPts.max(axis=0)+1])
    return iPts is not None and len(iPts)%2

def pointInPolygons2D(pts, plys, p):
    ios = [pointInPolygon2D(pts[ply], p) for ply in plys]
    return sum(ios)%2

def pointInTriHull3D(ABCs, p):
    ts = intersectTrianglesWithRay(ABCs, p, normVec(ABCs.max(axis=0).max(axis=0) - p))
    return len(ts)%2    

def arePointsCoplanar(pts):
    if len(pts) <= 3:
        return True
    u,v = normVec(pts[[1,2]] - pts[0])
    return np.all(np.abs(np.dot(normVec(pts[2:] - pts[0]), cross(u, v, True))) < dotEps)

def arePointsColinear(pts):
    if len(pts) == 2:
        return True
    if len(pts) == 3:
        u,v = normVec(pts[[1,2]] - pts[0])
        return np.abs(np.dot(u,v)) > 1-eps
    ds = normVec(pts[1:] - pts[0])
    return np.abs(np.dot(ds[1:], ds[0])).min() > 1-dotEps

def trianglesDoIntersect2D(t1, t2=None):
    if t2 is None:
        t1, t2 = t1
    for i in range(3):
        for j in range(3):
            if edgesIntersect2D(t1[i], t1[(i+1)%3], t2[j], t2[(j+1)%3]):
                return True
    for p in t2:
        if not pointInTriangle2D(t1[0], t1[1], t1[2], p):
            break
    else:
        return True
    for p in t1:
        if not pointInTriangle2D(t2[0], t2[1], t2[2], p):
            break
    else:
        return True
    return False

def intersectTrianglesWithEdge(ABCs, PQ, ns = None):
    ns = computeTriangleNormals(ABCs) if ns is None else ns
    sP = simpleSign(inner1d(PQ[0] - ABCs[:,0], ns))
    sQ = simpleSign(inner1d(PQ[1] - ABCs[:,0], ns))
    sMsk = sP != sQ
    if not sMsk.any():
        return False

    D, eLen = normVec(PQ[1]-PQ[0], True)
    ts = intersectTrianglesWithRay(ABCs[sMsk], PQ[0], D)
    return len(ts) and ts.min() < eLen

def intersectTrianglesWithRay(ABCs, O, D):
    e1s = ABCs[:,1] - ABCs[:,0]
    e2s = ABCs[:,2] - ABCs[:,0]
    hs = cross(D, e2s)
    dps = inner1d(e1s, hs)

    zMsk = np.abs(dps) > eps
    if not zMsk.any():
        return []
    fs = 1/dps[zMsk]
    ss = O - ABCs[zMsk,0]
    us = inner1d(ss, hs[zMsk]) * fs

    uMsk = np.bitwise_and(us >= 0, us <= 1)
    if not uMsk.any():
        return []
    qs = cross(ss[uMsk], e1s[zMsk][uMsk])
    vs = inner1d(D, qs) * fs[uMsk]

    vMsk = np.bitwise_and(vs >= 0, (us[uMsk] + vs) <= 1)
    if not vMsk.any():
        return []
    ts = inner1d(e2s[zMsk][uMsk][vMsk], qs[vMsk]) * fs[uMsk][vMsk]
    return ts[ts>eps]

# does not cover total inclusion
def polyhedraDoIntersect(aVerts, bVerts, aTris, bTris, aEdges, bEdges, aTrisNormals = None, bTrisNormals = None):
    #aEdges = facesToEdges(aFaces)
    #bTris = facesToTris(bFaces)
    for aEdge in aEdges:
        if intersectTrianglesWithEdge(bVerts[bTris], aVerts[aEdge], bTrisNormals):
            return True
    #bEdges = facesToEdges(bFaces)
    #aTris = facesToTris(aFaces)
    for bEdge in bEdges:
        if intersectTrianglesWithEdge(aVerts[aTris], bVerts[bEdge], aTrisNormals):
            return True
    return False

# https://github.com/erich666/jgt-code/blob/master/Volume_07/Number_2/Ganovelli2002/tet_a_tet.h
def tetrahedraDoIntersect(ptsA, ptsB, nsA, nsB):
    masks = np.zeros((4,4), np.bool8)
    coord = np.zeros((4,4), np.float32)

    def faceA(i):
        coord[i] = np.dot(ptsB - ptsA[i%3], nsA[i])
        masks[i] = coord[i] > 0
        return np.all(masks[i])

    def faceB(i):
        return np.all(np.dot(ptsA - ptsB[i%3], nsB[i]) > 0)

    def edge(f, g):
        if not np.all(np.bitwise_or(masks[f], masks[g])):
            return False

        for e in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]:
            if (masks[f,e[0]] and not masks[f,e[1]]) and (not masks[g,e[0]] and masks[g,e[1]]):
                if (coord[f,e[1]] * coord[g,e[0]] - coord[f,e[0]] * coord[g,e[1]]) > 0:
                    return False
            if (masks[f,e[1]] and not masks[f,e[0]]) and (not masks[g,e[1]] and masks[g,e[0]]):
                if (coord[f,e[1]] * coord[g,e[0]] - coord[f,e[0]] * coord[g,e[1]]) < 0:
                    return False

        return True

    def pointInside():
        return not np.all(np.any(masks, axis=0))

    fs = []
    for f in range(4):
        if faceA(f):
            return False
        for g in fs:
            if edge(f,g):
                return False
        fs.append(f)

    if pointInside():
        return True

    for f in range(4):
        if faceB(f):
            return False
        
    return True

def pyrasDoIntersect(ptsA, ptsB, nsA, nsB):
    masks = np.zeros((5,5), np.bool8)
    coord = np.zeros((5,5), np.float32)

    def faceA(i):
        # ptsA[i] only works with correct ns order!
        coord[i] = np.dot(ptsB - ptsA[i], nsA[i])
        masks[i] = coord[i] > 0
        return np.all(masks[i])

    def faceB(i):
        return np.all(np.dot(ptsA - ptsB[i], nsB[i]) > 0)

    def edge(f, g):
        if not np.all(np.bitwise_or(masks[f], masks[g])):
            return False

        for e in [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]:
            if (masks[f,e[0]] and not masks[f,e[1]]) and (not masks[g,e[0]] and masks[g,e[1]]):
                if (coord[f,e[1]] * coord[g,e[0]] - coord[f,e[0]] * coord[g,e[1]]) > 0:
                    return False
            if (masks[f,e[1]] and not masks[f,e[0]]) and (not masks[g,e[1]] and masks[g,e[0]]):
                if (coord[f,e[1]] * coord[g,e[0]] - coord[f,e[0]] * coord[g,e[1]]) < 0:
                    return False

        return True

    def pointInside():
        return not np.all(np.any(masks, axis=0))

    fs = []
    for f in [[0],[1],[0,1],[2],[1,2],[3],[2,3],[3,0],[4],[0,4],[1,4],[2,4],[3,4]]:
        if len(f) == 1:
            if faceA(f[0]):
                return False
        else:
            if edge(f[0],f[1]):
                return False

    if pointInside():
        return True

    for f in range(5):
        if faceB(f):
            return False

    return True

def intersectLinesLine2D(ons, o, n):
    ss = np.dot(ons[:,0] - o, n) / np.dot(ons[:,1], orthoVec(n))
    return ons[:,0] + orthoVec(ons[:,1]) * ss.reshape(-1,1)

def intersectEdgesEdge2D(es, e, withIdxs = False):
    eDir = e[1]-e[0]
    eOrt = orthoVec(eDir)

    s0 = simpleSign(np.dot(es[:,0] - e[0], eOrt))
    s1 = simpleSign(np.dot(es[:,1] - e[0], eOrt))
    cMsk = s0 != s1
    if not cMsk.any():
        return (None, None) if withIdxs else None

    eDirs = es[cMsk,1]-es[cMsk,0]
    ss = inner1d(e[0] - es[cMsk,0], orthoVec(eDirs)) / np.dot(eDirs, eOrt)
    iMsk = np.bitwise_and(ss >= 0, ss <= 1)

    if not iMsk.any():
        return (None, None) if withIdxs else None
    iPts = e[0] + eDir * ss[iMsk].reshape(-1,1)
    return (iPts, np.where(cMsk)[0][iMsk]) if withIdxs else iPts

def intersectEdgePlane(pts, o, n, sane=True):
    if not sane:
        ds = np.dot(pts - o, n)
        if np.all(ds < 0) or np.all(ds > 0) or np.any(ds == 0):
            return None
    v = normVec(pts[1] - pts[0])
    t = np.dot((o-pts[0]), n) / np.dot(v, n)
    return pts[0] + v * t

def intersectEdgesPlane(pts, o, n):
    vs = pts[:,1] - pts[:,0]
    ts = np.dot((o-pts[:,0]), n) / np.dot(vs, n)
    return pts[:,0] + vs * ts.reshape(-1,1)

def intersectThreePlanes(os, ns):
    return np.linalg.solve(ns, inner1d(os, ns)) #if np.linalg.det(ns) else np.linalg.lstsq(ns, inner1d(os,ns))[0]

def triangulatePoly2D(vs):
    tris = []
    poly = list(range(len(vs)))

    # check winding and flip for CW order
    if 0 > np.prod(vs * [[-1,1]] + np.roll(vs, -1, axis=0), axis=1).sum():
        poly = poly[::-1]

    idx = 0
    while len(poly) > 2:
        pdx, ndx = (idx-1)%len(poly), (idx+1)%len(poly)
        A, B, C = vs[poly[pdx]], vs[poly[idx]], vs[poly[ndx]]

        # check if concave or convex triangle
        if 0 < np.sign((B[0]-A[0]) * (C[1]-A[1]) - (B[1]-A[1]) * (C[0]-A[0])):
            idx = (idx+1)%len(poly)
            continue

        otherIdxs = [i for i in poly if i not in [poly[pdx], poly[idx], poly[ndx]]]
        for odx in otherIdxs:
            if pointInTriangle2D(A, B, C, vs[odx]):
                idx = (idx+1)%len(poly)
                break
        else:
            tris.append([poly[pdx], poly[idx], poly[ndx]])
            poly.pop(idx)
            idx %= len(poly)

    return np.int32(tris)

def triangulatePoly3D(vs):
    return triangulatePoly2D(aaProject3Dto2D(vs))

def aaProject3Dto2D(verts):
    vecs = normVec(verts - verts.mean(axis=0))
    eVals, eVecs = np.linalg.eigh(np.dot(vecs.T, vecs))
    pDim = np.abs(eVecs[:,eVals.argmin()]).argmax()
    pDir = np.eye(3)[pDim]
    pVerts = verts - pDir * np.dot(verts, pDir).reshape(-1,1)
    return pVerts[:, np.int32([pDim + 1, pDim + 2]) % 3]   

def ringNeighborElements(elements, vIdxs, n = 0):
    msk = np.zeros(len(elements), np.bool_)
    for vIdx in vIdxs:
        msk = np.bitwise_or(msk, np.any(elements == vIdx, axis=1))

    eMasked = elements[msk]
    return ringNeighborElements(elements, np.unique(eMasked.ravel()), n-1) if n else eMasked

def computeConvexPolygonVertexOrder(pts, refPt = None):
    cPt = pts.mean(axis=0)
    if refPt is not None:
        n = normVec(refPt - cPt)
        pts = projectPoints(pts, cPt, n, True)
        cPt = pts.mean(axis=0)
    dirs = normVec(pts - cPt)
    return np.argsort((np.arctan2(dirs[:,0], dirs[:,1]) + 2*np.pi) % (2*np.pi))

def findConnectedComponents(edges):
    comps = [set(edges[0])]
    for edge in edges[1:]:
        cIdxs = [cIdx for cIdx, comp in enumerate(comps) if not comp.isdisjoint(edge)]
        if not len(cIdxs):
            comps.append(set(edge))
        elif len(cIdxs) == 1:
            comps[cIdxs[0]].update(edge)
        elif cIdxs[0] != cIdxs[1]:
            comps[cIdxs[0]].update(comps.pop(cIdxs[1]))
    return comps

def findConnectedEdgeSegments(edges):
    segments = [[edge.tolist() if not type(edge) == list else edge] for edge in edges]
    while True:
        l = len(segments)
        for i, segmentA in enumerate(segments):
            for j, segmentB in enumerate(segments):
                if i == j:
                    continue
                if not set(flatten(segmentA)).isdisjoint(set(flatten(segmentB))):
                    segments[j] += segments[i]
                    segments.pop(i)
                    break
        if l == len(segments):
            break
    return segments

def appendUnique(lst, i):
    if i not in lst:
        lst.append(i)

def haveCommonElement(a, b):
    if len(b) < len(a):
        a, b = b, a
    for x in a:
        if x in b:
            return True
    return False

def rollRows(M, r):
    rows, cIdxs = np.ogrid[:M.shape[0], :M.shape[1]]
    r[r < 0] += M.shape[1]
    return M[rows, cIdxs - r[:, np.newaxis]]

def centerToUnitCube(pts):
    sPts = pts - pts.min(axis=0)
    sPts /= sPts.max()
    sPts -= sPts.max(axis=0)/2
    return sPts * 2    

def sortCounterClockwise(pts, c = None, returnOrder = False):
    c = pts.mean(axis=0) if c is None else c
    d = normVec(pts-c)
    #angles = np.arccos(d[:,0])
    #m = d[:,1] < 0
    #angles[m] = 2*np.pi-angles[m]
    angles = (np.arctan2(d[:,0], d[:,1]) + 2*np.pi) % (2*np.pi)
    order = np.argsort(angles)
    return order if returnOrder else pts[order]

def flipTrianglesUniformly(ts, vs = None):
    edgeHashToTriIdxs = {}
    triEdgeHashs = cantorPiV(facesToEdges(ts, False)).reshape(-1,3)
    for i, teHashs in enumerate(triEdgeHashs):
        for teh in teHashs:
            if teh not in edgeHashToTriIdxs:
                edgeHashToTriIdxs[teh] = [i]
            else:
                edgeHashToTriIdxs[teh].append(i)

    visited = np.zeros(len(ts), dtype=np.bool_)
    visited[0] = True
    queue = [0]
    
    while len(queue):
        tIdx = queue.pop(0)
        tri = ts[tIdx]
        tehs = triEdgeHashs[tIdx]
        for j, teh in enumerate(tehs):
            nIdxs = edgeHashToTriIdxs[teh]

            for nIdx in nIdxs:
                if nIdx == tIdx or visited[nIdx]:
                    continue
                nTri = ts[nIdx]
                if (tri[j], tri[(j+1)%3]) in [(nTri[0], nTri[1]), (nTri[1], nTri[2]), (nTri[2], nTri[0])]:
                    ts[nIdx] = nTri[::-1]
                    triEdgeHashs[nIdx] = triEdgeHashs[nIdx][[1,0,2]]

                visited[nIdx] = True
                queue.append(nIdx)

        if not len(queue) and not visited.all():
            queue.append(visited.argmin())         

    if vs is not None:
        vz = np.concatenate([vs, [vs.mean(axis=0)]])
        tz = padHor(ts, len(vs))
        if computeTetraVolumes(vz[tz]).sum() > 0:
            ts = ts[:,::-1]            
                
    return ts

def filterForUniqueEdges(edges):
    edges = np.int32(edges) if type(edges) == list else edges
    uHashs, uIdxs = np.unique(cantorPiV(edges), return_index=True)
    return edges[uIdxs]

def filterForSingleEdges(edges):
    hashs = cantorPiV(edges)
    uHashs, uIdxs, uCounts = np.unique(hashs, return_index=True, return_counts=True)
    return np.array([edges[idx] for idx, count in zip(uIdxs, uCounts) if count == 1])

def reIndexIndices(arr):
    uIdxs = np.unique(flatten(arr) if type(arr) == list else arr.ravel())
    reIdx = np.zeros(uIdxs.max()+1, np.int32)
    reIdx[uIdxs] = np.argsort(uIdxs)
    return [reIdx[ar] for ar in arr] if type(arr) == list else reIdx[arr]

def computePolygonCentroid2D(pts, withArea=False):
    rPts = np.roll(pts, 1, axis=0)
    w = pts[:,0] * rPts[:,1] - rPts[:,0] * pts[:,1]
    area = w.sum() / 2.0
    if area:
        centroid = np.sum((pts + rPts) * w.reshape(-1,1), axis=0) / (6 * area)
    else:
        centroid = pts.mean(axis=0)
    return (centroid, np.abs(area)) if withArea else centroid

def computeTriangleAngles(pts):
    return computePolygonAngles(pts)

def computeTrianglesAngles(ptss):
    ds = normVec(np.roll(ptss, -1, axis=1) - ptss)
    return np.arccos(np.clip(inner1d(np.vstack(ds), np.vstack(-np.roll(ds, 1, axis=1))), -1, 1)).reshape(-1,3)

def computePolygonAngles(pts):
    ds = normVec(np.roll(pts, -1, axis=0) - pts)
    return np.arccos(np.clip(inner1d(ds, -np.roll(ds, 1, axis=0)), -1, 1))

def computeTriangleNormal(pts, normed = True):
    AB, AC = pts[1:] - pts[0] if pts.ndim < 3 else (pts[:,1:] - pts[:,0].reshape(-1,1,3)).T
    return cross(AB, AC, normed)

def computeTriangleNormals(pts, normed = True):
    ABAC = pts[:,1:] - pts[:,0].reshape(-1,1,3)
    return cross(ABAC[:,0], ABAC[:,1], normed)

def computeTriangleArea(pts, signed = True):
    if pts.shape[1] == 2:
        area = simpleDet3x3(pad2Dto3D(pts, 1).T)/2
        return area if signed else np.abs(area)
    else:
        return norm(computeTriangleNormal(pts, False))/2

def computeTriangleAreas(pts, signed = True):
    if pts.shape[-1] == 2:
        areas = simpleDets3x3(np.transpose(np.pad(pts, [[0,0],[0,0],[0,1]], mode='constant',constant_values = 1), axes=[0,2,1]))/2
        return areas if signed else np.abs(areas)
    else:
        return norm(computeTriangleNormals(pts, False))/2

def computeTriangleEdgeCenters(pts):
    return (pts + np.roll(pts, 1, axis=0))/2

def computeTriangleEdgeCenterss(ptss):
    return (ptss + np.roll(ptss, 1, axis=1))/ 2

def computeTriangleGradients(pts):
    dirs, lens = normVec(np.roll(pts, 1, axis=0) - np.roll(pts, -1, axis=0), True)
    s = lens.sum()/2
    area = np.sqrt(s * np.prod(s-lens))
    #area = computeTriangleArea(pts, False)
    return np.dot(dirs, Mr2D(np.pi/2)) * lens.reshape(-1,1) / (2 * area)

def computeBaryWeights(tVerts, pt):
    nDim = len(pt)
    tVertss = np.repeat([tVerts], nDim+1, axis=0)
    rng = np.arange(nDim+1)
    tVertss[rng, rng] = pt
    ws = computeTriangleAreas(tVertss) if nDim == 2 else computeTetraVolumes(tVertss)
    return ws / ws.sum()

def computeBaryWeightss(tVertss, pt):
    nDim = len(pt)
    tIdxs = np.arange(len(tVertss)*(nDim+1)).reshape(-1, nDim+1).ravel()
    vIdxs = np.arange(len(tVertss)*(nDim+1)) % (nDim+1)
    tVertss = np.repeat(tVertss, nDim+1, axis=0)
    tVertss[tIdxs, vIdxs] = pt    
    ws = (computeTriangleAreas(tVertss) if nDim == 2 else computeTetraVolumes(tVertss)).reshape(-1, nDim+1)
    return ws / ws.sum(axis=1).reshape(-1,1)

def computeGaussianCurvatures(vs, ts): # per vertex
    angles = computeTrianglesAngles(vs[ts])
    return (np.pi * 2) - np.bincount(ts.ravel(), weights = angles.ravel())

def computeDihedralAngles(vs, ts, returnUnique = True): # per edge, only closed manifolds
    edges = facesToEdges(ts, False)
    eHashs = cantorPiV(edges)
    uHashs, uIdxs = np.unique(eHashs, return_index=True)
    eFaceIdxs = np.repeat(np.arange(len(ts)), 3)[np.argsort(eHashs)].reshape(-1,2)

    triNormals = computeTriangleNormals(vs[ts])
    perEdgeAngles = np.arccos(np.clip(inner1d(triNormals[eFaceIdxs[:,0]], triNormals[eFaceIdxs[:,1]]), -1, 1))
    return (edges[uIdxs], perEdgeAngles) if returnUnique else (edges, uIdxs, perEdgeAngles)

def computeSummedEdgeAngles(vs, ts): # per vertex, only closed manifolds
    edges, uIdxs, perEdgeAngles = computeDihedralAngles(vs, ts, False)
    return np.bincount(edges[uIdxs].ravel(), weights = np.repeat(perEdgeAngles, 2))    

def filterForManifoldness(vs, ts):
    edges = facesToEdges(ts, False)
    eHashs = cantorPiV(edges)
    uHashs, uCts = np.unique(eHashs, return_counts=True)

    teMap = eHashs.reshape(-1,3)
    tMsk = np.ones(len(ts), np.bool_)
    for singleEdgeHash in uHashs[uCts == 1]:
        tMsk[np.any(teMap == singleEdgeHash, axis=1)] = False

    if not tMsk.any():
        warnings.warn('No manifold extracted, mesh probably open.')
        return None

    if tMsk.all():
        return vs, ts

    mts = ts[tMsk]
    return filterForManifoldness(vs[np.unique(mts.ravel())], reIndexIndices(mts))

def computeTetraVolume(pts, signed = True):
    a, b, c = pts[1:] - pts[0]
    vol = np.dot(cross(a,b), c) / 6.0
    return vol if signed else np.abs(vol)

def computeTetraVolumes(ptss, signed = True):
    abcs = ptss[:,1:] - ptss[:,0].reshape(-1,1,3)
    vols = inner1d(cross(abcs[:,0], abcs[:,1]), abcs[:,2]) / 6.0
    return vols if signed else np.abs(vols)

def computeTetraEdgeCenters(pts):
    return np.vstack([pts[0] + pts[1:], pts[1:] + np.roll(pts[1:], 1, axis=0)])/2

def computeTetraEdgeCenterss(ptss):
    return np.concatenate([ptss[:,0,np.newaxis] + ptss[:,1:], ptss[:,1:] + np.roll(ptss[:,1:], 1, axis=1)], axis=1)/2

def computeTetraTriCenters(pts):
    return np.dot(1 - np.eye(4), pts)/3

def computeTetraTriCenterss(ptss):
    return dotAxBs(1 - np.eye(4), ptss)/3

def computeTetraGradients(pts):
    tris = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
    triNormals = computeTriangleNormals(pts[tris])
    dps = inner1d(triNormals, pts[tris].mean(axis=1) - pts)
    gradDirs = triNormals * -simpleSign(dps).reshape(-1,1)
    gradDirs *= computeTriangleAreas(pts[tris], False).reshape(-1,1)
    return gradDirs / (3 * computeTetraVolume(pts, False))

def computePolyVolume(pts, faces):
    tris = facesToTris(faces) # works with convex faces only
    center = pts[np.unique(tris)].mean(axis=0)
    return np.sum([computeTetraVolume(np.vstack([center, pts[tri]]), False) for tri in tris])

def computeHexaVolume(pts):
    return computePolyVolume(pts[[0,2,6,3,1,4,7,5]], sixCubeFaces)
    #a, b, c = pts[[1,2,3]] - pts[0]
    #d, e, f = pts[[4,5,6]] - pts[7]
    #return (np.abs(np.dot(cross(a,b), c)) + np.abs(np.dot(cross(d,e), f))) / 2

def computeHexaVolumes(ptss):
    return np.sum([computeHexaVolume(pts) for pts in ptss])
    #abcs = ptss[:,[1,2,3]] - ptss[:,0].reshape(-1,1,3)
    #defs = ptss[:,[4,5,6]] - ptss[:,7].reshape(-1,1,3)
    #return (np.abs(inner1d(cross(abcs[:,0], abcs[:,1]), abcs[:,2])) + np.abs(inner1d(cross(defs[:,0], defs[:,1]), defs[:,2])))/2

def computePolyhedronCentroid(vertices, faces, returnVolume=False):
    tris = facesToTris(faces)
    tets = padHor(tris, -1)
    verts = np.vstack([vertices, [vertices[np.unique(tris)].mean(axis=0)]])
    tetPts = verts[tets]
    tetCentroids = tetPts.mean(axis=1)
    tetVolumes = computeTetraVolumes(tetPts, False)
    tetVolumesSum = tetVolumes.sum()
    polyCentroid = np.dot(tetVolumes/tetVolumesSum, tetCentroids) if tetVolumesSum > eps else tetCentroids.mean(axis=0)
    return (polyCentroid, tetVolumesSum) if returnVolume else polyCentroid

def computePrincipalStress(sMat):
    eVals, eVecs = np.linalg.eigh(sMat)
    eVals = np.abs(eVals)
    o = np.argsort(eVals)[::-1]
    return eVecs.T[o], eVals[o]

def projectPoints(pts, o, n, return2d=False):
    vs = pts - o
    ds = np.dot(vs, n)
    projected = pts - ds.reshape(-1,1) * n
    if not return2d:
        return projected

    up = np.float32([0,0,1])
    x = cross(up, n, True)
    theta = np.arccos(np.dot(up, n))
    A = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    R = np.eye(3) + np.sin(theta) * A + (1-np.cos(theta)) * np.dot(A,A)
    return np.dot(R.T, projected.T).T[:,:2]

def concatPolyParts(polyParts):
    polys = []
    for part in polyParts:
        for i, poly in enumerate(polys):
            if norm(part[-1] - poly[0]) < eps:
                polys[i] = np.vstack([part, poly])
                break
            if norm(part[0] - poly[0]) < eps:
                polys[i] = np.vstack([part[::-1], poly])
                break
            if norm(part[0] - poly[-1]) < eps:
                polys[i] = np.vstack([poly, part])
                break
            if norm(part[-1] - poly[-1]) < eps:
                polys[i] = np.vstack([poly, part[::-1]])
                break
        else:
            polys.append(part)
    return polys

def limitedDissolve2D(verts):
    vIdxs = []
    n = len(verts)
    for vIdx in range(n):
        pIdx = (vIdx - 1) % n
        nIdx = (vIdx + 1) % n
        if vIdx == nIdx or norm(verts[vIdx] - verts[nIdx]) < eps:
            continue
        vecs = normVec(verts[[pIdx, nIdx]] - verts[vIdx])
        if np.abs(np.dot(vecs[0], vecs[1])) < (1 - eps):
            vIdxs.append(vIdx)
    return limitedDissolve2D(verts[vIdxs]) if len(vIdxs) < n else verts[vIdxs]

def resamplePoly(verts, nTarget, isClosed = True):
    vs = np.vstack([verts, verts[0]]) if isClosed else verts
    csLength = np.pad(np.cumsum(norm(np.diff(vs, axis=0))), (1,0))
    sDists = np.linspace(0, csLength[-1], nTarget + isClosed)
    rvs = np.transpose([np.interp(sDists, csLength, v) for v in vs.T])
    return rvs[:-1] if isClosed else rvs

def interpolateBezierTriangle(points, normals):
    b = lambda i, j: (2*points[i] + points[j] - np.dot(points[j]-points[i], normals[i]) * normals[i])/3
    bs = np.array([points[0], points[1], points[2], b(0,1), b(1,0), b(1,2), b(2,1), b(2,0), b(0,2)])
    bezierCenter = bs[3:].sum(axis=0)/4 - bs[:3].sum(axis=0)/6
    return bezierCenter, normVec(bezierCenter - points.mean(axis=0))

def quaternionSlerp(qa, qb, t):
    # http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/
    ratios = [1, 0]
    cosHalfTheta = np.dot(qa, qb)
    if abs(cosHalfTheta) < 1:
        halfTheta = np.arccos(cosHalfTheta)
        sinHalfTheta = np.sqrt(1 - cosHalfTheta**2)
        if abs(sinHalfTheta) < eps:
            ratios = [0.5, 0.5]
        else:
            ratios = np.sin(np.float32([1-t, t]) * halfTheta) / sinHalfTheta
    return np.dot(ratios, [qa, qb])

def quaternionAverage(quats, weights = None):
    # https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions    
    weights = np.ones(len(quats),np.float32) if weights is None else weights
    Q = quats * (weights/weights.sum()).reshape(-1,1)
    eigVals, eigVecs = np.linalg.eigh(np.dot(Q.T, Q))
    return eigVecs[:,np.argmax(eigVals)]

def rotateAsToB(As, Bup, Aup = np.float32([0,0,1])):
    x = cross(Aup,Bup,True)
    theta = np.arccos(np.dot(Aup,Bup)/(np.linalg.norm(Aup)*np.linalg.norm(Bup)))
    Mx = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    R = np.eye(3) + np.sin(theta) * Mx + (1-np.cos(theta))*np.dot(Mx,Mx)
    return np.dot(R,As.T).T

def orthogonalizeOrientations(Ss):
    U,s,Vt = np.linalg.svd(Ss)
    #U, Vt = simpleSVDs(Ss, False) # experimental
    #R = U @ Vt
    #R[simpleDets3x3(R) < 0, -1] *= -1
    #return R
    Vt[:,-1] *= simpleDets(U @ Vt)[:,None]
    return U @ Vt

def computeRotationAngle(M):
    return min(np.float32([np.arctan2(M[0,0], M[0,1]),np.arctan2(M[0,1],M[0,0])]) % (np.pi/2))

def computeRotationAngles(Ms):
    a0 = np.arctan2(Ms[:,0,0], Ms[:,0,1]) % (np.pi/2)
    a1 = np.arctan2(Ms[:,0,1], Ms[:,0,0]) % (np.pi/2)
    return np.minimum(a0, a1)

def computeMinEulerAngle(A, B):
    return alignBtoA(A, B, True)    

def computeMinEulerAngles(A, Bs):
    return np.float32([alignBtoA(A, B, True) for B in Bs])

def alignBtoA(A, B, angleOnly = False):
    nDim = A.shape[0]
    if nDim == 2:
        os = [[0,1],[1,0]]
        ss = [[1,1],[1,-1],[-1,1],[-1,-1]]
    else:
        os = [[0,1,2],[1,2,0],[2,0,1],[0,2,1],[2,1,0],[1,0,2]]
        ss = [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]]
    Bs = np.tile(B[os],[len(ss),1,1]) * np.repeat(ss, len(os), axis=0).reshape(-1,nDim,1)
    a = np.arccos(np.clip((innerAxBs(A,Bs).sum(axis=1)-1)/2,-1,1))
    return a.min() if angleOnly else Bs[np.argmin(a)]                

def alignBstoA(A, Bs):
    return np.float32([alignBtoA(A, B) for B in Bs])

def computeMinTransformation(M):
    if len(M) == 2:
        return Mr2D(np.arctan2(M[0,1], M[0,0]) % (np.pi/2)).T
    argmax3 = lambda v: 0 if v[0] > max(v[1],v[2]+eps) else (1 if v[1] > v[2]+eps else 2)
    aM = np.abs(M)
    o0 = [0,1,2]
    o = [o0.pop(argmax3(aM[:,0]))] + (o0 if aM[o0[0],1] > aM[o0[1],1] else o0[::-1])
    N = np.zeros((3,3))
    N[[0,1,2],o] = np.sign(M[o,[0,1,2]])
    return np.dot(N, M)

def computeMinTransformations(Ms):
    if len(Ms[0]) == 2:
        return np.float32([computeMinTransformation(M) for M in Ms])
    n = len(Ms)
    nRange = np.arange(n)
    aMs = np.abs(Ms)
    fst = np.argmax(aMs[:,:,0], axis=1)
    rst = (np.tile([[1,2]],[n,1]) + fst.reshape(-1,1)) % 3
    idxs = np.argmax(aMs[np.transpose([nRange]*2), rst, np.ones((n,2),np.int32)], axis=1)

    rs = np.transpose([fst, rst[nRange,idxs], rst[nRange,idxs-1]])

    z = np.zeros_like(Ms)
    idxsA = np.repeat(nRange, 3).reshape(-1,3)
    idxsB = (np.arange(n*3)%3).reshape(-1,3)
    z[idxsA, idxsB, rs] = np.sign(Ms[idxsA, rs, idxsB])

    return inner3x3M(z, Ms)

def computeAvgTransformation(Ms):
    if Ms.shape[1] == 2:
        U,D,V = np.linalg.svd(Ms.sum(axis=0))
        M = np.dot(U, V)
        return computeMinTransformation(M)
    M = np.float32([computeAvgDirection(Ms[:,i]) for i in range(3)])
    M[1:] -= M[0] * inner1d(M[1:], M[0]).reshape(-1,1)
    U,D,V = np.linalg.svd(M)
    return np.dot(U,V)

def orthogonalizeMatrix(M):
    q,r = np.linalg.qr(M) # may produce sign flips
    return np.copysign(q, M)

def computeWeightedTransformation(Ms, ws = None):
    ws = np.ones(len(Ms), np.float32) if ws is None else ws
    rMs = alignBstoA(Ms[ws.argmax()], Ms) * (ws/ws.sum()).reshape(-1,1,1) 
    return orthogonalizeMatrix(rMs.sum(axis=0))
    
def computeAvgDirection(vs, maxIter = 100, tol = 1e-6):
    avgDir = normVec(np.random.rand(3))

    for i in range(maxIter):
        newAvgDir = normVec(np.sum(vs * np.dot(vs, avgDir).reshape(-1,1), axis=0))
        if norm(avgDir - newAvgDir) < tol:
            break
        avgDir = newAvgDir

    return avgDir

def computeJacobian(pts, scaled = False):
    return computeJacobians(pts.reshape(1, pts.shape[0], pts.shape[1]), scaled)[0]

def computeJacobians(ptss, scaled = False):
    if ptss.shape[2] == 2:
        if ptss.shape[1] == 3: # tri
            Js = np.float32([ptss[:,[2,1]] - ptss[:,0,None],
                             ptss[:,[0,2]] - ptss[:,1,None],
                             ptss[:,[1,0]] - ptss[:,2,None]])
        else: # quad
            Js = np.float32([ptss[:,[3,1]] - ptss[:,0,None],
                             ptss[:,[0,2]] - ptss[:,1,None],
                             ptss[:,[1,3]] - ptss[:,2,None],
                             ptss[:,[2,0]] - ptss[:,3,None]])
        return simpleDets2x2(np.vstack(np.transpose(normVec(Js) if scaled else Js, axes=[1,0,2,3]))).reshape(-1, ptss.shape[1])
    else:
        if ptss.shape[1] == 4: # tet
            Js = np.float32([ptss[:,[2,3,1]] - ptss[:,0,None],
                             ptss[:,[0,3,2]] - ptss[:,1,None],
                             ptss[:,[0,1,3]] - ptss[:,2,None],
                             ptss[:,[2,1,0]] - ptss[:,3,None]])
        else: # hex (diagonal order)
            Js = np.float32([ptss[:,[2,1,3]] - ptss[:,0,None],
                             ptss[:,[4,5,0]] - ptss[:,1,None],
                             ptss[:,[6,4,0]] - ptss[:,2,None],
                             ptss[:,[0,5,6]] - ptss[:,3,None],
                             ptss[:,[7,1,2]] - ptss[:,4,None],
                             ptss[:,[1,7,3]] - ptss[:,5,None],
                             ptss[:,[3,7,2]] - ptss[:,6,None],
                             ptss[:,[5,4,6]] - ptss[:,7,None]])
        return simpleDets3x3(np.vstack(np.transpose(normVec(Js) if scaled else Js, axes=[1,0,2,3]))).reshape(-1, ptss.shape[1])

def vecsToOrthoMatrix(vs): # experimental
    if len(vs) > 3:
        vBuckets = [[],[],[]]
        for v in vs:
            for b in vBuckets:
                if not b:
                    b.append(v)
                    break
                if np.arccos(np.clip(np.abs(np.dot(b[0], v)),0,1)) < np.pi/8:
                    b.append(v)
                    break
        vs = np.float32([computeAvgDirection(b) for b in vBuckets])
    return orthogonalizeMatrix(normVec(vs))

def computeIsoContours(vs, ts, ss, isoValue, t = eps):
    vIoMsk = ss > isoValue
    vXoMsk = np.abs(ss - isoValue) < t
    tIoCnt = vIoMsk[ts].sum(axis=1)
    tMsk = np.bitwise_and(tIoCnt > 0, tIoCnt < 3)
    #tMsk = np.bitwise_or(tMsk, vXoMsk[ts].any(axis=1))

    class EdgeCollector:
        verts, edges, vertKeyToIdx, edge = [], [], {}, []

        def addVertex(self, v, vKey):
            if vKey in self.vertKeyToIdx.keys():
                self.edge.append(self.vertKeyToIdx[vKey])
            else:
                self.edge.append(len(self.verts))
                self.vertKeyToIdx[vKey] = len(self.verts)
                self.verts.append(v)

        def flush(self):
            if len(self.edge) == 2 and self.edge[0] != self.edge[1]:
                self.edges.append(self.edge)
            self.edge = []

        def getVertsAndEdges(self):
            return np.vstack(self.verts), np.vstack(groupEdges(self.edges))

    ec = EdgeCollector()
    usedEdgesHashs = []
    for tri in ts[tMsk]:
        xMsk = vXoMsk[tri]
        if xMsk.sum() == 1: # vertex on contour
            e = tri[np.bitwise_not(xMsk)]
            bCoords = 1-np.abs(isoValue - ss[e]) / np.abs(ss[e[1]] - ss[e[0]])
            if np.any(bCoords < 0) or np.any(bCoords > 1) or np.abs(bCoords).sum() - 1 > eps:
                continue
            vIdx = tri[xMsk][0]
            ec.addVertex(vs[vIdx], -vIdx)
            ec.addVertex(np.dot(bCoords, vs[e]), cantorPi(e[0], e[1]))
        elif xMsk.sum() == 2: # edge on contour
            vIdx, vJdx = tri[xMsk]
            eHash = cantorPi(vIdx, vJdx)
            if eHash in usedEdgesHashs:
                continue
            usedEdgesHashs.append(eHash)
            ec.addVertex(vs[vIdx], -vIdx)
            ec.addVertex(vs[vJdx], -vJdx)
        else: # nothing on contour
            es = faceToEdges(tri)
            es = es[vIoMsk[es].sum(axis=1) == 1]
            bCoords = 1 - np.abs(isoValue - ss[es]) / np.abs(ss[es[:,1]] - ss[es[:,0]]).reshape(-1,1)
            edgeVerts = innerVxM(bCoords, np.transpose(vs[es], axes=[0,2,1]))
            for eVert, eHash in zip(edgeVerts, cantorPiV(es)):
                ec.addVertex(eVert, eHash)
        ec.flush()

    return ec.getVertsAndEdges()
