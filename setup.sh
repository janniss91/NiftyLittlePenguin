# Install poetry if it is not on the system
curl -sSL https://install.python-poetry.org | python3 -
# pip3 install poetry

# TODO: This should only be done inside the container so must go to the dockerfile.
# Add a soft link so that poetry can find the python3 interpreter.
ln -s /usr/bin/python3 /usr/bin/python

# Make sure poetry is on the path.
export PATH="/root/.local/bin:$PATH"

# Install dependencies
poetry update
poetry install