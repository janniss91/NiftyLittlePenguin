FROM ubuntu:20.04

# https://jdhao.github.io/2021/01/17/install_python3_in_ubuntu_docker/
RUN apt-get update -y && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt curl

WORKDIR /nlp
COPY . .

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    # Add a soft link so that poetry can find the python3 interpreter.
    ln -s /usr/bin/python3 /usr/bin/python && \
    # Make sure poetry is on the path and install dependencies
    export PATH="/root/.local/bin:$PATH" && \
    poetry install --without dev

ENTRYPOINT [ "/bin/bash", "entrypoint.sh" ]
