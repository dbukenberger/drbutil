import os
import numpy as np
from .util import *

def loadObjFile(fileName, extraTags = []):
    verts = []
    vertsNormals = []
    faces = []
    edges = []
    extras = [[] for eTag in extraTags]
    with open(fileName, 'r') as fh:
        for line in fh.readlines():
            lTag = line.split(' ')[0]

            if lTag == 'v':
                verts.append(list(map(float, line.strip().split(' ')[1:])))

            if lTag == 'vn':
                vertsNormals.append(list(map(float, line.strip().split(' ')[1:])))

            if lTag == 'f':
                if '//' in line:
                    faces.append(list(map(lambda x: int(x.split('//')[0])-1, line.strip().split(' ')[1:])))
                elif '/' in line:
                    faces.append(list(map(lambda x: int(x.split('/')[0])-1, line.strip().split(' ')[1:])))
                else:
                    faces.append(list(map(lambda x: int(x)-1, line.strip().split(' ')[1:])))

            if lTag == 'l':
                edges.append(list(map(lambda x: int(x)-1, line.strip().split(' ')[1:])))

            for e, eTag in enumerate(extraTags):
                if lTag == eTag:
                    ln = line.strip().split(' ')[1:]
                    if len(ln) == 1:
                        extras[e].append(float(ln[0]))
                    else:
                        extras[e].append(list(map(float, ln)))

    returnList = []
    if verts:
        returnList.append(np.float32(verts))
    if vertsNormals:
        returnList.append(np.float32(vertsNormals))
    if faces:
        fLens = list(map(len, faces))
        if min(fLens) == max(fLens):
            faces = np.int32(faces)
        returnList.append(faces)
    if edges:
        returnList.append(np.int32(edges))
    if extras:
        returnList += list(map(np.float32, extras))

    return returnList

def loadMeshFile(fileName):
    verts = []
    tets = []
    hexas = []
    with open(fileName, 'r') as fh:
        mode = None
        skipOne = False
        for line in fh.readlines():
            line = line.strip()

            if line == 'End':
                break

            if 'Vertices' in line:
                mode = 'verts'
                skipOne = len(line.split(' ')) == 1
                continue

            if 'Tetrahedra' in line:
                mode = 'tet'
                skipOne = len(line.split(' ')) == 1
                continue

            if 'Hexahedra' in line:
                mode = 'hex'
                skipOne = len(line.split(' ')) == 1
                continue

            if 'Quadrilaterals' in line:
                mode = 'quads'
                skipOne = len(line.split(' ')) == 1
                continue

            if mode in [None, 'quads']:
                continue

            if skipOne:
                skipOne = False
                continue
            
            if mode == 'verts':
                verts.append([c for c in line.strip().split(' ')[:-1] if c])

            if mode == 'tet':
                tets.append([c for c in line.strip().split(' ')[:-1] if c])

            if mode == 'hex':
                hexas.append([c for c in line.strip().split(' ')[:-1] if c])

    res = [np.float32(verts)]
    if len(tets):
        res.append(np.int32(tets)-1)
    if len(hexas):
        res.append(np.int32(hexas)-1)

    return res

def writeObjFile(filePath, vertices, faces, edges=[], comment='', subTags=[], roundFloats = True):
    with open(filePath, 'w') as fh:
        fh.write('# obj export\n')
        if len(comment):
            fh.write('# %s\n'%comment)
        if not len(subTags):
            fh.write('o %s\n' % os.path.basename(filePath)[:-4])
            for vertex in vertices:
                if roundFloats:
                    fh.write('v %0.6f %0.6f %0.6f\n' % tuple(vertex))
                else:
                    fh.write('v ' + ' '.join(map(str, vertex)) + '\n')
            for face in faces:
                fh.write(('f' + ' %d' * len(face) + '\n') % tuple(face + 1))
            for edge in edges:
                fh.write('l %d %d\n' % tuple(edge + 1))
        else:
            vOffset = 0
            for subIdx, tag in enumerate(subTags):
                fh.write('o %s\n' % tag)
                for vertex in vertices[subIdx]:
                    if roundFloats:
                        fh.write('v %0.6f %0.6f %0.6f\n' % tuple(vertex))
                    else:
                        fh.write('v ' + ' '.join(map(str, vertex)) + '\n')
                if len(faces):
                    for face in map(np.int32, faces[subIdx]):
                        fh.write(('f' + ' %d' * len(face) + '\n') % tuple(face + vOffset + 1))
                if len(edges):
                    for edge in map(np.int32, edges[subIdx]):
                        fh.write('l %d %d\n' % tuple(edge + vOffset + 1))
                vOffset += len(vertices[subIdx])

def writeOvmFile(filePath, vertices, faces, cells):
    with open(filePath, 'w') as fh:
        fh.write('OVM ASCII\n')

        fh.write('Vertices\n%d\n'%len(vertices))
        for vertex in vertices:
            fh.write('%0.6f %0.6f %0.6f\n' % tuple(vertex))

        edges = []
        edgesHashs = []
        halfFaces = []
        for face in faces:
            hFace = []
            fEdges = faceToEdges(face)
            for fEdge in fEdges:
                eHashO = cantorPiO(fEdge[0], fEdge[1])
                eHashR = cantorPiO(fEdge[1], fEdge[0])
                if eHashO in edgesHashs:
                    hFace.append(edgesHashs.index(eHashO) * 2)
                elif eHashR in edgesHashs:
                    hFace.append(edgesHashs.index(eHashR) * 2 + 1)
                else:
                    hFace.append(len(edges) * 2)
                    edges.append(fEdge)
                    edgesHashs.append(eHashO)
            halfFaces.append(hFace)

        fh.write('Edges\n%d\n'%len(edges))
        for edge in edges:
            fh.write('%d %d\n' % tuple(edge[::-1]))

        polyFaces = []
        facesHashs = []
        polyhedra = []
        for cell in cells:
            polyhedron = []
            for fIdx in cell:
                face = faces[fIdx]
                face.sort()
                fHash = cantorPiK(list(map(int, face)))
                if fHash in facesHashs:
                    polyhedron.append(facesHashs.index(fHash) * 2 + 1)
                else:
                    polyhedron.append(len(polyFaces) * 2)
                    facesHashs.append(fHash)
                    polyFaces.append(halfFaces[fIdx])
            polyhedra.append(polyhedron)       
        
        fh.write('Faces\n%d\n'%len(polyFaces))
        for face in polyFaces:
            fh.write((str(len(face)) + ' %d' * len(face) + '\n') % tuple(face[::-1]))

        fh.write('Polyhedra\n%d\n'%len(polyhedra))
        for cell in polyhedra:
            fh.write((str(len(cell)) + ' %d' * len(cell) + '\n') % tuple(cell))

def writeTxtFile(filePath, data):
    with open(filePath, 'w') as fh:
        for line in data:
            fmt = '%f' if type(line[0]) == float else '%d'
            fh.write(' '.join([fmt%e for e in line]) + '\n')

def writePlyFile(filePath, vertices, faces = [], edges = [], verticesColors = [], verticesNormals = [], withAln = False, cmnt = None):
    with open(filePath, 'w') as fh:
        fh.write('ply\nformat ascii 1.0\ncomment %s\n'%os.path.basename(filePath)[:-4])
        if cmnt is not None:
            fh.write('comment %s\n'%cmnt)
        fh.write('element vertex %d\n'%len(vertices))

        vertProps = vertices
        props = ['x', 'y', 'z']
        if len(verticesColors):
            vertProps = np.hstack([vertProps, verticesColors])
            props += ['red', 'green', 'blue']
        if len(verticesNormals):
            vertProps = np.hstack([vertProps, verticesNormals])
            props += ['nx', 'ny', 'nz']

        fh.write(''.join(['property float %s\n'%s for s in props]))
        if len(faces):
            fh.write('element face %d\n'%len(faces))
            fh.write('property list uchar uint vertex_indices\n')
        if len(edges):
            fh.write('element edge %d\n'%len(edges))
            fh.write('property int vertex1\nproperty int vertex2\n')

        fh.write('end_header\n')
        
        for vertProp in vertProps:
            fh.write(' '.join(['%0.6f'%p for p in vertProp])+'\n')
            #fh.write(' '.join(['%g'%p for p in vertProp])+'\n')
        for face in faces:
            fh.write((str(len(face))+' %d'*len(face)+'\n')%tuple(face))
        for edge in edges:
            fh.write('%d %d\n'%tuple(edge))

    if withAln:
        with open(filePath[:-3] + 'aln', 'w') as fh:
            fh.write('1\n%s\n#\n'%os.path.basename(filePath))
            fh.write('\n'.join([str(r)[1:-1] for r in np.eye(4)]))
            fh.write('\n0\n')

def loadPlyFile(fileName):
    vertices = []
    faces = []
    with open(fileName, 'r') as fh:
        header = True
        for line in fh.readlines():
            line = line.strip()
            if 'end_header' in line:
                header = False
                continue
            if header:
                if 'element vertex' in line:
                    nVerts = int(line.split(' ')[-1])
                if 'element face' in line:
                    nFaces = int(line.split(' ')[-1])
                continue

            elif len(line):
                if nVerts:
                    vertices.append(np.float32(line.split(' ')))
                    nVerts -= 1
                elif nFaces:
                    faces.append(np.int32(line.split(' ')[1:]))
                    nFaces -= 1
    verts = np.float32(vertices)
    fLens = list(map(len, faces))
    if min(fLens) == max(fLens):
        faces = np.int32(faces)
    if verts.shape[1] == 4:
        return verts[:,:-1], verts[:,-1], faces
    elif verts.shape[1] == 6:
        return verts[:,:3], verts[:,3:], faces
    else:
        return verts, faces

def writeMeshFile(filePath, vertices, hexas): # hex
    with open(filePath, 'w') as fh:
        fh.write('MeshVersionFormatted 1\nDimension 3\n')

        fh.write('Vertices\n%d\n'%len(vertices))
        for vertex in vertices:
            fh.write('%0.6f %0.6f %0.6f 0\n' % tuple(vertex))

        fh.write('Hexahedra\n%d\n'%len(hexas))
        for hexa in hexas:
            fh.write(('%d ' * 8 + '0\n') % tuple(hexa+1))

        fh.write('End\n')

def writeMshFile(filePath, vertices, tets): # tet
    with open(filePath, 'w') as fh:
        fh.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n%d\n'%len(vertices))
        for i, vertex in enumerate(vertices):
            fh.write('%d %0.6f %0.6f %0.6f\n'%(i+1, vertex[0], vertex[1], vertex[2]))
        fh.write('$EndNodes\n$Elements\n%d\n'%len(tets))
        for i,tet in enumerate(tets+1):
            fh.write('%d 4 2 0 0 %d %d %d %d\n'%(i+1, tet[0], tet[1], tet[2], tet[3]))
        fh.write('$EndElements\n')

def writeStressFile(fileName, verts, cells, forceIdxs, forceVecs, fixedIdxs, stress):
    with open(fileName, 'w') as fh:
        fh.write('Version 2.0\n')

        nDim = verts.shape[1]
        cVal = cells.shape[1]
        
        if nDim == 2:
            fh.write('Plane %s 1\n'%('Tri' if cVal == 3 else 'Quad'))
        else:
            fh.write('Solid %s 1\n'%('Tet' if cVal == 4 else 'Hex'))

        fh.write('Vertices: %d\n'%len(verts))
        vLine = ' '.join(['%g']*nDim)+'\n'
        for vert in verts:
            fh.write(vLine%tuple(vert))

        fh.write('Elements: %d\n'%len(cells))
        cLine = ' '.join(['%d']*cVal)+'\n'
        for cell in cells:
            fh.write(cLine%tuple(cell+1))

        fh.write('Node Forces: %d\n'%len(forceIdxs))
        vLine = ' '.join(['%g']*nDim)+'\n'
        for fIdx, fVec in zip(forceIdxs, forceVecs):
            fh.write('%d '%(fIdx+1) + vLine%tuple(fVec))

        fh.write('Fixed Nodes: %d\n'%len(fixedIdxs))
        for fIdx in fixedIdxs:
            fh.write('%d\n'%(fIdx+1))

        fh.write('Cartesian Stress: %d\n'%len(stress))
        sLine = ' '.join(['%g']*(nDim-1)*3)+'\n'
        for s in stress:
            fh.write(sLine%tuple(s))

def loadStressFile(fileName):
    verts = []
    cells = []
    forceIdxs = []
    forceVecs = []
    fixedIdxs = []
    stress = []
    with open(fileName, 'r') as fh:
        mode = None
        skipOne = True
        for line in fh.readlines():
            line = line.strip()

            if ':' in line:
                mode, num = line.split(':')
                continue

            if mode == 'Vertices':
                verts.append(line.split(' '))

            if mode == 'Elements':
                cells.append(line.split(' '))

            if mode == 'Node Forces':
                forceIdxs.append(line.split(' ')[0])
                forceVecs.append(line.split(' ')[1:])

            if mode == 'Fixed Nodes':
                fixedIdxs.append(line)

            if mode == 'Cartesian Stress':
                stress.append(line.split(' '))

    verts = np.float32(verts)
    cells = np.int32(cells)-1
    forceIdxs = np.int32(forceIdxs)-1
    forceVecs = np.float32(forceVecs)
    fixedIdxs = np.int32(fixedIdxs)-1
    stress = np.float32(stress)
    if stress.shape[1] == 6:
        nDim = 3
        sMats = np.empty((nDim, nDim, len(stress)), np.float32)        
        s00, s11, s22, s12, s02, s01 = stress.T
        vmStress = np.sqrt(((s00-s11)**2 + (s11-s22)**2 + (s22-s00)**2 + 6*(s01**2+s12**2+s02**2))/2)
        sMats[0,0] = s00
        sMats[[0,1],[1,0]] = s01
        sMats[1,1] = s11
        sMats[[0,2],[2,0]] = s02
        sMats[[1,2],[2,1]] = s12
        sMats[2,2] = s22
    else:
        nDim = 2
        sMats = np.empty((nDim, nDim, len(stress)), np.float32)        
        s00, s11, s01 = stress.T
        vmStress = np.sqrt(s00**2 + s11**2 - s00*s11 + 3*s01**2)
        sMats[0,0] = s00
        sMats[[0,1],[1,0]] = s01
        sMats[1,1] = s11

    sMats = np.transpose(sMats, axes = [2,0,1])

    pStress = np.zeros_like(sMats)
    pStressE = np.zeros((len(stress), nDim), np.float32)
    for i, sMat in enumerate(sMats):
        pStress[i], pStressE[i] = computePrincipalStress(sMat)

    return verts, cells, forceIdxs, forceVecs, fixedIdxs, vmStress, pStress, pStressE, sMats

def loadVtkFile(fileName):
    verts = []
    hexas = []
    with open(fileName, 'r') as fh:
        mode = None
        for line in fh.readlines():
            lTag = line.split(' ')[0]

            if lTag == 'POINTS':
                mode = 'pts'
                continue

            if lTag == 'CELLS':
                mode = 'cls'
                continue

            if lTag == 'CELL_TYPES':
                mode = None
                continue

            if mode is None:
                continue
            
            if mode == 'pts':
                verts.append(line.strip().split(' '))

            if mode == 'cls':
                hexas.append(line[1:].strip().split(' '))
                
    return np.float32(verts), np.int32(hexas)

def loadHybridFile(fileName, withHexFlags = False):
    with open(fileName, 'r') as fh:
        line = fh.readline().strip()
        numVertices, numFaces, numPolyhedra = np.int32(line.split(' '))

        vertices = np.empty((numVertices, 3), np.float32)
        for i in range(numVertices):
            line = fh.readline().strip()
            vertices[i] = [float(v) for v in line.split(' ')]
            
        faces = []
        for i in range(numFaces):
            line = fh.readline().strip()
            face = np.int32(line.split(' '))
            faces.append(face[1:])
            
        polyhedra = []
        hexFlags = []
        for i in range(numPolyhedra):
            line = fh.readline().strip()
            if len(line) == 1 and withHexFlags:
                    hexFlags.append('1' in line)
            elif len(line):
                polyhedron = np.int32(line.split(' '))
                polyhedra.append(polyhedron[1:])
                line = fh.readline() # skip orientation line
            else:
                break

    return (vertices, faces, polyhedra, np.bool_(hexFlags)) if withHexFlags else (vertices, faces, polyhedra)
