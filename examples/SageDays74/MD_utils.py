import numpy as np
from numba import double, jit
import k3d
   

@jit((double[:,:], double[:], double, double))
def lennardjones(U, box, sigma = 0.3405, epsilon=0.9959):

    ndim = len(U)
    npart = len(U[0])

    F = np.zeros((ndim, npart))

    Epot = 0.0
    Vir = 0.0
   
    C = epsilon/sigma**2*48 
    
    rc2 = 6.25
    rc2i=1.0/rc2
    rc2i *= sigma**2
    rc6i=rc2i*rc2i*rc2i
    ecut=rc6i*(rc6i-1.0)
    
    for i in range(npart):
        for j in range(npart):
            if i > j:
                X  = U[0, j] - U[0, i]
                Y  = U[1, j] - U[1, i]
                Z  = U[2, j] - U[2, i]

                # Periodic boundary condition
                X  -= box[0] * np.rint(X/box[0])
                Y  -= box[1] * np.rint(Y/box[1])
                Z  -= box[2] * np.rint(Z/box[2])

                # Distance squared
                r2 = X*X + Y*Y + Z*Z
                if(r2 < rc2):
                    r2i = 1.0 / r2
                    r2i *= sigma**2 # use nm scale
                    r6i = r2i*r2i*r2i
                    Epot = Epot + r6i*(r6i-1.0) - ecut

                    ftmp = C * r6i*(r6i- 0.5) 
                    Vir += ftmp

                    ftmp *= r2i

                    F[0, i] -= ftmp * X
                    F[1, i] -= ftmp * Y
                    F[2, i] -= ftmp * Z
                    F[0, j] += ftmp * X
                    F[1, j] += ftmp * Y
                    F[2, j] += ftmp * Z
    Epot = Epot * 4.0 * epsilon

    return Epot, F, Vir




class simple_molecule_vis(object):
    
    @classmethod
    def box_coords(cls, bs = 2.2):
        a = bs/2.0
        box =   np.array([-a,-a,-a] + [-a,a,-a] +[a,a,-a] + [a,-a,-a]+ [-a,-a,-a]+\
                                   [-a,-a,a] + [-a,a,a] +[a,a,a] + [a,-a,a]+ [-a,-a,a]+\
                                    [-a,a,a]+[-a,a,-a]+[-a,a,a]+\
                                   [a,a,a]+[a,a,-a]+[a,a,a]+\
                                   [a,-a,a]+[a,-a,-a]+[a,-a,a]) 
        return box


    def update_box(self, bs = 1):
        self.box.vertices = self.box_coords(bs=bs)

    def __init__(self,bs=1.0):
        self.new_plot(bs=bs)
        
    def new_plot(self,bs=1.0):
        points_number = 1
        positions = 50 * np.random.random_sample((points_number,3)) - 25
        colors = np.random.randint(0, 0x777777, points_number)

        self.plot = k3d.plot()
        self.pkts = k3d.points(positions, colors, point_size=.3)
        self.plot += self.pkts
        self.plot.camera_auto_fit = False
        self.plot.grid_auto_fit = False
        self.box = k3d.line(self.box_coords(bs=bs))
      
        self.plot += self.box
    def __repr__(self):
        self.plot.display()
        return "K3D fast molecule viewer"



