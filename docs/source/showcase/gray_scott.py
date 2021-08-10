import numpy as np
import k3d
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    plot = k3d.plot(screenshot_scale=1.0, camera_auto_fit=False)
    headless = k3d_remote(plot, get_headless_driver(), width=320, height=226)

    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    size = (512, 512)

    block_size = 128

    nx = (size[0] // block_size) * block_size
    ny = size[1]
    blocks = nx * ny // block_size

    u = np.ones((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)

    texture = k3d.texture(attribute=u, color_range=[0.0, 0.35],
                          color_map=k3d.matplotlib_color_maps.Viridis_r)
    plot.camera = [0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 1]
    plot += texture

    texture.color_map = k3d.colormaps.matplotlib_color_maps.Viridis_r

    # Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065 # Bacteria 1
    # Du, Dv, F, k = 0.14, 0.06, 0.035, 0.065 # Bacteria 2
    Du, Dv, F, k = 0.16, 0.08, 0.060, 0.062  # Coral
    # Du, Dv, F, k = 0.19, 0.05, 0.060, 0.062 # Fingerprint
    # Du, Dv, F, k = 0.10, 0.10, 0.018, 0.050 # Spirals
    # Du, Dv, F, k = 0.12, 0.08, 0.020, 0.050 # Spirals Dense
    # Du, Dv, F, k = 0.10, 0.16, 0.020, 0.050 # Spirals Fast
    # Du, Dv, F, k = 0.16, 0.08, 0.020, 0.055 # Unstable
    # Du, Dv, F, k = 0.16, 0.08, 0.050, 0.065 # Worms 1
    # Du, Dv, F, k = 0.16, 0.08, 0.054, 0.063 # Worms 2
    # Du, Dv, F, k = 0.16, 0.08, 0.035, 0.060 # Zebrafish

    dt = 0.5 * 1

    pars = {'nx': nx,
            'ny': ny,
            'Du': Du,
            'Dv': Dv,
            'dt': dt,
            'F': F,
            'k': k}

    src = """
        __device__ inline float laplace2d(int idx, float *a)
        {{
          return(a[idx-1] + a[idx+1] + a[idx-{nx}] + a[idx+{nx}] - 4.0f * a[idx] );
        }}
    
      
        __global__ void iterate_RDS(float *a,float *da,float *b,float *db)
        {{
          int idx = blockDim.x*blockIdx.x + threadIdx.x;
          float k = {k}f;  
          float F = {F}f;
          
          if(idx<{nx} || idx>{nx}*({ny}-1)-1) {{
              return;
          }}
          
          int x = idx % {nx};
          
          if(x==0 || x=={nx}-1) {{
              return;
          }}
          
          float u = a[idx]; 
          float v = b[idx];       
         
          da[idx] = u + {dt}f*(  -u*v*v + F*(1.0f-u) + {Du}*laplace2d(idx, a));
          db[idx] = v + {dt}f*(   u*v*v - (F+k)*v + {Dv}*laplace2d(idx, b));
        }}
        """.format(**pars)

    mod = SourceModule(src)
    RDSv = mod.get_function("iterate_RDS")

    r = 20
    u[ny // 2 - r:ny // 2 + r, nx // 2 - r:nx // 2 + r] = 0.50
    v[ny // 2 - r:ny // 2 + r, nx // 2 - r:nx // 2 + r] = 0.25

    u += 0.05 * np.random.random((ny, nx))
    v += 0.05 * np.random.random((ny, nx))

    u_g = gpuarray.to_gpu(u)
    du_g = gpuarray.empty_like(u_g)

    v_g = gpuarray.to_gpu(v)
    dv_g = gpuarray.empty_like(v_g)

    for i in range(30000):
        RDSv(u_g, du_g, v_g, dv_g, block=(block_size, 1, 1), grid=(blocks, 1))
        RDSv(du_g, u_g, dv_g, v_g, block=(block_size, 1, 1), grid=(blocks, 1))

        if (i + 1) % 500 == 0:
            v = v_g.get()
            texture.attribute = v

    v = v_g.get()
    texture.attribute = v

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1.0)

    ctx.pop()
    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
