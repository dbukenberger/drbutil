import numpy as np


# basic geometry and utility functions

try:
    from numpy.core.umath_tests import inner1d
except ImportError:
    def inner1d(u, v): return np.einsum('ij,ij->i', u, v)

def normVec(v):
    if v.ndim == 1:
        n = np.sqrt(np.dot(v, v))
        return v / n if n else v * 0
    else:
        n = np.sqrt(inner1d(v, v))
        m = n != 0
        v = v.copy()
        v[m] /= n[m].reshape(-1, 1)
        return v

def norm(v): return np.sqrt(np.dot(v, v) if v.ndim == 1 else inner1d(v, v))

def normZeroToOne(data): return (np.float32(data) - np.min(data))/(np.max(data) - np.min(data)) if len(data) else data

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

def getPrincipalStress(sMat):
    eVals, eVecs = np.linalg.eig(sMat)
    eVals = np.abs(eVals)
    o = np.argsort(eVals)[::-1]
    return eVecs.T[o], eVals[o]

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

def generatePointsOnCircle(n, in3D = False):
    if n<=0:
        return None
    offset = np.pi/n
    alpha = np.linspace(offset, np.pi*2-offset, n)
    pts = np.transpose([np.cos(alpha), np.sin(alpha)])
    return np.pad(pts, [[0,0],[0,1]], 'constant', constant_values = 0) if in3D else pts

def distPointToEdge(A, B, P):
    AtoB = B - A
    AtoP = P - A
    BtoP = P - B
    
    if np.dot(AtoB, AtoP) <= 0:
        return norm(AtoP)
    elif np.dot(-AtoB, BtoP) <= 0:
        return norm(BtoP)
    else:
        d = normVec(AtoB)
        return norm(AtoP-np.dot(AtoP,d)*d)

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
    d = (D[1] - C[1]) * (B[0] - A[0]) - (D[0] - C[0]) * (B[1] - A[1])
    u = (D[0] - C[0]) * (A[1] - C[1]) - (D[1] - C[1]) * (A[0] - C[0])
    v = (B[0] - A[0]) * (A[1] - C[1]) - (B[1] - A[1]) * (A[0] - C[0])
    if d < 0:
        u, v, d = -u, -v, -d
    return (0 <= u <= d) and (0 <= v <= d)

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

def pointInTriangle2D(A, B, C, P):
    v0, v1, v2 = C - A, B - A, P - A
    u, v, d = v2[1] * v0[0] - v2[0] * v0[1], v1[1] * v2[0] - v1[0] * v2[1], v1[1] * v0[0] - v1[0] * v0[1]
    if d < 0:
        u, v, d = -u, -v, -d
    return u >= 0 and v >= 0 and (u + v) <= d

"""
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
"""

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
    eVals, eVecs = np.linalg.eig(np.dot(vecs.T, vecs))
    pDim = np.abs(eVecs[:,eVals.argmin()]).argmax()
    pDir = np.eye(3)[pDim]
    pVerts = verts - pDir * np.dot(verts, pDir).reshape(-1,1)
    return pVerts[:, np.int32([pDim + 1, pDim + 2]) % 3]   

def getConvexPolygonVertexOrder(pts, refPt = None):
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

def filterForUniqueEdges(edges):
    edges = np.int32(edges) if type(edges) == list else edges
    uHashs, uIdxs = np.unique(cantorPiV(edges), return_index=True)
    return edges[uIdxs]

def filterForSingleEdges(edges):
    hashs = cantorPiV(edges)
    uHashs, uIdxs, uCounts = np.unique(hashs, return_index=True, return_counts=True)
    return np.array([edges[idx] for idx, count in zip(uIdxs, uCounts) if count == 1])

def reIndexIndices(arr):
    uIdxs = np.unique(flatten(arr))
    reIdx = np.zeros(uIdxs.max()+1, np.int32)
    reIdx[uIdxs] = np.argsort(uIdxs)
    return [reIdx[ar] for ar in arr] if type(arr) == list else reIdx[arr]

def computePolygonCentroid2D(pts, withArea=False):
    rPts = np.roll(pts, 1, axis=0)
    w = pts[:,0] * rPts[:,1] - rPts[:,0] * pts[:,1]
    area = w.sum() / 2.0
    centroid = np.sum((pts + rPts) * w.reshape(-1,1), axis=0) / (6 * area)
    return (centroid, np.abs(area)) if withArea else centroid

def getTriangleNormal(pts, normed = True):
    AB, AC = pts[1:] - pts[0] if pts.ndim < 3 else (pts[:,1:] - pts[:,0].reshape(-1,1,3)).T
    return cross(AB, AC, normed)

def getTriangleNormals(pts, normed = True):
    ABAC = pts[:,1:] - pts[:,0].reshape(-1,1,3)
    return cross(ABAC[:,0], ABAC[:,1], normed)

def getTriangleArea(pts):
    if pts.shape[1] == 2:
        return simpleDet3x3(pad2Dto3D(pts, 1).T)/2
    else:
        return norm(getTriangleNormal(pts, False))/2

def getTriangleAreas(pts, signed = True):
    if pts.shape[-1] == 2:
        areas = simpleDets3x3(np.transpose(np.pad(pts, [[0,0],[0,0],[0,1]], mode='constant',constant_values = 1), axes=[0,2,1]))/2
        return areas if signed else np.abs(areas)
    else:
        return norm(getTriangleNormals(pts, False))/2

def computeTetraVolume(pts):
    a, b, c = pts[1:] - pts[0]
    return np.abs(np.dot(cross(a,b), c) / 6.0)

def computeTetraVolumes(ptss):
    abcs = ptss[:,1:] - ptss[:,0].reshape(-1,1,3)
    return np.abs(inner1d(cross(abcs[:,0], abcs[:,1]), abcs[:,2]) / 6.0)

def computePolyVolume(pts, faces):
    tris = facesToTris(faces) # works with convex faces only
    center = pts[np.unique(tris)].mean(axis=0)
    return np.sum([computeTetraVolume(np.vstack([center, pts[tri]])) for tri in tris])

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
    #return np.sum([computeHexaVolume(pts) for pts in ptss])

def computePolyhedronCentroid(vertices, faces, returnVolume=False):
    tris = facesToTris(faces)
    tets = padHor(tris, -1)
    verts = np.vstack([vertices, [vertices[np.unique(tris)].mean(axis=0)]])
    tetPts = verts[tets]
    tetCentroids = tetPts.mean(axis=1)
    tetVolumes = computeTetraVolumes(tetPts)
    tetVolumesSum = tetVolumes.sum()
    polyCentroid = np.dot(tetVolumes/tetVolumesSum, tetCentroids) if tetVolumesSum > eps else tetCentroids.mean(axis=0)
    return (polyCentroid, tetVolumesSum) if returnVolume else polyCentroid

def getConvexPolygonVertexOrder(pts, refPt=None):
    cPt = pts.mean(axis=0)
    if refPt is not None:
        n = normVec(refPt - cPt)
        pts = projectPoints(pts, cPt, n, True)
        cPt = pts.mean(axis=0)
    dirs = normVec(pts - cPt)
    return np.argsort((np.arctan2(dirs[:,0], dirs[:,1]) + 2 * np.pi) % (2 * np.pi))

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

def getFaceCutIdxs(faceMasks):
    faceMasksCat = np.concatenate(faceMasks)
    faceLensCum = np.cumsum([0]+list(map(len, faceMasks)))
    firstLastIdxs = np.transpose([faceLensCum[:-1], faceLensCum[1:]-1])

    firstLasts = faceMasksCat[firstLastIdxs.ravel()].reshape(-1,2)
    trueStartIdxs = faceLensCum[np.where(firstLasts.sum(axis=1) == 1)[0]]

    firstLastIdxsFlat = firstLasts.ravel()[:-1]
    falseStartIdxs = faceLensCum[np.where(np.reshape(firstLastIdxsFlat[1:] ^ firstLastIdxsFlat[:-1], (-1,2))[:,1])[0]+1]

    inFaceIdxs = np.where(faceMasksCat[:-1] ^ faceMasksCat[1:])[0]+1

    resIdxs = np.int32(sorted(set(inFaceIdxs).difference(falseStartIdxs).union(trueStartIdxs)))
    return resIdxs.reshape(-1,2) - np.reshape(faceLensCum[:-1], (-1,1))

def interpolateBezierTriangle(points, normals):
    b = lambda i, j: (2*points[i] + points[j] - np.dot(points[j]-points[i], normals[i]) * normals[i])/3
    bs = np.array([points[0], points[1], points[2], b(0,1), b(1,0), b(1,2), b(2,1), b(2,0), b(0,2)])
    bezierCenter = bs[3:].sum(axis=0)/4 - bs[:3].sum(axis=0)/6
    return bezierCenter, normVec(bezierCenter - points.mean(axis=0))

def rotateAsToB(As, Bup, Aup = np.float32([0,0,1])):
    x = cross(Aup,Bup,True)
    theta = np.arccos(np.dot(Aup,Bup)/(np.linalg.norm(Aup)*np.linalg.norm(Bup)))
    Mx = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    R = np.eye(3) + np.sin(theta) * Mx + (1-np.cos(theta))*np.dot(Mx,Mx)
    return np.dot(R,As.T).T

def getRotationMatrixXYZ(alpha = 0, beta = 0, gamma = 0):
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    M = np.dot(Rx, np.dot(Ry, Rz))
    return M

def getMat(ab,cd):
    dp = np.dot(ab,cd.T)
    return np.argsort(np.abs(dp)) * simpleSign(dp)

def mixFaceDirections(vds, vns, fn, v = False):
    gs = np.zeros(6, np.float32)
    for vIdx in range(3):
        #vrs = normVec(projectPoints(vds[vIdx], np.float32([0,0,0]), fn))
        vrs = rotateAsToB(vds[vIdx], fn, vns[vIdx])
        vrs = rotateAsToB(vrs, np.float32([0,0,1]), fn)

        g = np.arctan2(vrs[0,1], vrs[0,0]) % (np.pi/2)
        gs[vIdx] = g
        gs[vIdx+3] = g - np.pi/2

    M = np.float32([[1,1,1,0,0,0],
                    [1,1,0,0,0,1],
                    [1,0,1,0,1,0],
                    [1,0,0,0,1,1],
                    [0,1,1,1,0,0],
                    [0,1,0,1,0,1],
                    [0,0,1,1,1,0],
                    [0,0,0,1,1,1]])

    r = np.dot(M/3, gs)
    d = gs - r.reshape(-1,1)
    ds = np.sum((d * M)**2, axis=1)
    gamma = r[np.argmin(ds)]
    
    #return rotateAsToB(Mr3D(0,0,gamma/3)[:-1], fn, np.float32([0,0,1]))
    return rotateAsToB(Mr3D(0,0,gamma)[:,:-1].T, fn, np.float32([0,0,1]))
