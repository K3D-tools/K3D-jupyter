FROM python:3.9.16-slim

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update
RUN apt-get install -y -qq curl
RUN apt-get install -y libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 wget

# google chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN dpkg --install google-chrome-stable_current_amd64.deb || apt-get -f install -y


RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
RUN nvm install v16
RUN npm install -g webpack webpack-cli

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install pytest pixelmatch flask selenium webdriver-manager chromedriver-binary scikit-image vtk

WORKDIR /opt/app/src

CMD ["/bin/bash"]
