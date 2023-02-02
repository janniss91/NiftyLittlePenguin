# NiftyLittlePenguin

## Build

### Local Build

If you don't have Python 3.8 installed, use pyenv to install it.

    sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

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

    # Download poetry.
    curl -sSL https://install.python-poetry.org | python3 -

    # Add a soft link so that poetry can find the python3 interpreter.
    ln -s /usr/bin/python3 /usr/bin/python

    # Make sure poetry is on the path and install dependencies
    export PATH="/root/.local/bin:$PATH"
    poetry update
    poetry install

### Docker Build

To build the project using docker, run the following command:

    make build_container

## Run

### Run Locally

### Run in Docker Container

To run the project using docker, run the following command:

    make run_container
