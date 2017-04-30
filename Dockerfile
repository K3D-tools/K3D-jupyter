FROM andrewosh/binder-base

USER main

RUN pip install jupyter-pip ipywidgets
RUN pip install k3d
RUN pip install numba

RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
RUN jupyter nbextension enable --py --sys-prefix k3d

RUN $HOME/anaconda2/envs/python3/bin/pip install jupyter-pip ipywidgets
RUN $HOME/anaconda2/envs/python3/bin/pip install k3d
RUN $HOME/anaconda2/envs/python3/bin/pip install numba

RUN mkdir -p $HOME/.jupyter
RUN echo "c.NotebookApp.token = ''" >> $HOME/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password=''" >> $HOME/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password_required=False" >> $HOME/.jupyter/jupyter_notebook_config.py