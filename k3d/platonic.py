import numpy as np
from itertools import product
import k3d


class Dodecahedron:
    
    def __init__(self, origin, size=1):
        origin = np.array(origin, dtype=np.float32)
        
        if origin.shape == (3,):
            fi = (1 + np.sqrt(5))/2
            dodecahedron_vertices = list(product([-1,1],[-1,1],[-1,1])) 
            dodecahedron_vertices += [(0,fi,1/fi),(0,-fi,1/fi),(0,-fi,-1/fi),(0,fi,-1/fi),
                   (1/fi,0,fi),(-1/fi,0,fi),(-1/fi,0,-fi),(1/fi,0,-fi),
                   (fi,1/fi,0),(-fi,1/fi,0),(-fi,-1/fi,0),(fi,-1/fi,0)]
            dodecahedron_vertices = np.float32(size * np.array(dodecahedron_vertices) + origin)
            
            self.vertices = dodecahedron_vertices
            self.indices = [0,1,18, 0,1,10, 1,9,10, 0,10,14, 10,14,15, 4,10,15, 4,9,10, 4,5,9, 4,5,19, 4,15,19, 6,15,19, 6,16,19,
                   6,7,16, 6,7,8, 6,8,11, 2,3,17, 2,3,8, 2,8,11, 1,3,13, 1,3,18,  3,17,18, 1,9,13, 9,12,13, 5,9,12,
                   5,12,19, 12,16,19, 7,12,16, 3,7,8, 3,7,12, 3,12,13, 14,6,15, 14,6,11, 2,11,14, 0,17,18, 0,2,17, 0,2,14]
        else:
            raise TypeError('Origin should have 3 coordinates.')
    
    def mesh(self):
        return k3d.mesh(self.vertices, self.indices)


class Cube:
    
    def __init__(self, origin, size=1):
        origin = np.array(origin, dtype=np.float32)
        
        if origin.shape == (3,):
            cube_vertices = np.array(list(product([1,-1], [1, -1], [1, -1])), np.float32)
            cube_vertices = np.float32(size * cube_vertices + origin)
            
            self.vertices = cube_vertices
            self.indices = [0,1,2, 1,2,3, 0,1,4, 1,4,5, 1,3,5, 3,5,7, 0,2,4, 2,4,6, 2,3,7, 2,6,7, 4,5,6, 5,6,7]
        
        else:
            raise TypeError('Origin should have 3 coordinates.')
    
    def mesh(self):
        return k3d.mesh(self.vertices, self.indices)


class Icosahedron:
    
    def __init__(self, origin, size=1):
        origin = np.array(origin, dtype=np.float32)
        
        if origin.shape == (3,):
            fi = (1 + np.sqrt(5))/2
            icosahedron_vertices = [(1,fi,0), (1,-fi,0), (-1,fi,0), (-1,-fi,0),
                                    (fi,0,1), (fi,0,-1), (-fi,0,1), (-fi,0,-1),
                                    (0,1,fi), (0,1,-fi), (0,-1,fi), (0,-1,-fi)]
            icosahedron_vertices = np.float32(size * np.array(icosahedron_vertices) + origin)
            
            self.vertices = icosahedron_vertices
            self.indices = [0,2,8, 0,4,8, 0,2,9, 0,5,9, 2,6,8,2,7,9, 2,6,7, 0,4,5, 1,4,5, 1,5,11,
                            7,9,11, 3,7,11, 3,6,7, 3,6,10, 4,8,10,6,8,10, 1,4,10, 1,3,11, 1,3,10, 5,9,11]
        
        else:
            raise TypeError('Origin should have 3 coordinates.')
    
    def mesh(self):
        return k3d.mesh(self.vertices, self.indices)


class Octahedron:
    
    def __init__(self, origin, size=1):
        origin = np.array(origin, dtype=np.float32)
        
        if origin.shape == (3,):
            octahedron_vertices = [(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1)]
            octahedron_vertices = np.float32(size * np.array(octahedron_vertices) + origin)
            
            self.vertices = octahedron_vertices
            self.indices = [0,1,2, 0,1,5, 1,2,3, 1,3,5,0,4,5, 0,2,4, 2,3,4, 3,4,5]
        
        else:
            raise TypeError('Origin should have 3 coordinates.')
    
    def mesh(self):
        return k3d.mesh(self.vertices, self.indices)
    
    
class Tetrahedron:
    
    def __init__(self, origin, size=1):
        origin = np.array(origin, dtype=np.float32)
        
        if origin.shape == (3,):
            tetrahedron_vertices = [(1, 1, 1),(1, -1, -1),(-1, 1, -1),(-1, -1, 1)]
            tetrahedron_vertices = np.float32(size * np.array(tetrahedron_vertices) + origin)
        
            self.vertices = tetrahedron_vertices
            self.indices = [0,1,2, 0,1,3, 1,2,3, 0,2,3]
        
        else:
            raise TypeError('Origin should have 3 coordinates.')
    
    def mesh(self):
        return k3d.mesh(self.vertices, self.indices)
