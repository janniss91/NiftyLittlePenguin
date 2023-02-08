# NiftyLittlePenguin

## Build

### Local Build

If you don't have Python 3.8 installed, use pyenv to install it.

    sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl

    curl https://pyenv.run | bash

    # You will be asked to run the following lines after installing pyenv.
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    # Restart your terminal afterwards:
    exec "$SHELL"

    # Then install Python 3.8.2
    pyenv install -v 3.8.2
    
    # Set pyenv to use Python 3.8.2
    pyenv global 3.8.2

Then download poetry and install the dependencies.

    # Download and install poetry.
    curl -sSL https://install.python-poetry.org | python3 -

    # Make sure poetry is on the path by either typing:
    export PATH="/root/.local/bin:$PATH"

    # ... or typing, depending on what poetry asks you to do:
    export PATH="/home/your_username/.local/bin:$PATH"

    poetry install

### Docker Build

To build the project using docker, run the following command:

    make build_container

## Run

### Run Experimental Notebooks

You can run any of the experimental notebooks from the `notebooks` directory in jupyterlab.

To start jupyterlab, run:

    jupyter lab

Add the niftlittlepenguin kernel to jupyterlab to have access to all dependencies:

    poetry run python3 -m ipykernel install --user --name niftylittlepenguin --display-name "niftylittlepenguin"

### Run Locally

### Run in Docker Container

To run the project using docker, run the following command:

    make run_container
