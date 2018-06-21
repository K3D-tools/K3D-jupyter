FROM jupyter/scipy-notebook:1fbaef522f17

USER jovyan

RUN git clone https://github.com/K3D-tools/K3D-jupyter.git

WORKDIR K3D-jupyter

RUN pip install jupyter-pip ipywidgets
RUN pip install k3d

RUN jupyter nbextension install --py --user widgetsnbextension
RUN jupyter nbextension install --py --user k3d

RUN jupyter nbextension enable --py --user widgetsnbextension
RUN jupyter nbextension enable --py --user k3d

RUN mkdir -p $HOME/.jupyter
RUN echo "c.NotebookApp.token = ''" >> $HOME/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password=''" >> $HOME/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password_required=False" >> $HOME/.jupyter/jupyter_notebook_config.py