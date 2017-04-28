FROM andrewosh/binder-base

USER main

RUN git clone https://github.com/K3D-tools/K3D-jupyter.git

WORKDIR K3D-jupyter

RUN pip install jupyter-pip ipywidgets
RUN pip install k3d
RUN pip install numba

RUN $HOME/anaconda2/envs/python3/bin/pip install jupyter-pip ipywidgets
RUN $HOME/anaconda2/envs/python3/bin/pip install k3d
RUN $HOME/anaconda2/envs/python3/bin/pip install numba