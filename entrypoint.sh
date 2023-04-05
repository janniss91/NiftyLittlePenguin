echo "The container is up and running."
echo "Python version: $(python --version)"

# This line is used to make sure that poetry is on the $PATH.
export PATH="/root/.local/bin:$PATH"

poetry run uvicorn inference_service.serve:app --proxy-headers --host "0.0.0.0" --port 80

# Connect to a container shell. Uncomment to develop directly in the container.
# /bin/bash
