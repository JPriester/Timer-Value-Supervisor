FROM python:3.11-slim-buster
# Install required system packages
RUN apt-get update && apt-get install -y git
# Set up a non-root user (replace 'vscode' with your desired username)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME
# Create a directory for your project and set it as the working directory
WORKDIR /workspace
# Switch to the non-root user
USER $USERNAME
# Upgrade pip and install Python packages
RUN pip install --upgrade pip
# Copy the requirements.txt file into the container
COPY requirements.txt .

# Check if CUDA is available
RUN if command -v nvidia-smi > /dev/null 2>&1; then \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
else \
    pip3 install torch torchvision torchaudio; \
fi
RUN pip install --user -r requirements.txt
# test the install below
RUN pip install --user 'git+https://github.com/adysonmaia/sb3-plus#egg=sb3-plus'
# Set the default command to open a shell (optional)
CMD ["/bin/bash"]
