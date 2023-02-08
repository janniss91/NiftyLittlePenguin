echo "The container is up and running."
echo "Python version: $(python --version)"

# This line is used to make sure that poetry is on the $PATH.
export PATH="/root/.local/bin:$PATH"

# Connect to a container shell.
/bin/bash
