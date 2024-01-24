FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Add the current user to the image, to use the same user as the host

ARG USER_NAME=user
ARG USER_ID=1000
ARG GROUP_NAME=users
ARG GROUP_ID=1000

# fix for tzdata asking for user input
ARG DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Update the system and install required packages
RUN apt-get update && apt-get install -y sudo git apt-utils htop ffmpeg libsm6 libxext6

RUN groupadd --gid $GROUP_ID $GROUP_NAME
RUN useradd --uid $USER_ID --gid $GROUP_ID --shell /bin/bash --create-home $USER_NAME

# don't require password with sudo, for convenience
# not the safest thing to do, but hopefully okay inside the container
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Add the new user to the sudo group
RUN usermod -aG sudo $USER_NAME

# install conda stuff still as root
RUN conda update -n base conda
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN conda install -c bottler nvidiacub -y
RUN conda install pytorch3d -c pytorch3d -y

# Set the default user to the new user
USER $USER_NAME

WORKDIR /code

ENV PATH="$PATH:/home/$USER_NAME/.local/bin"
RUN pip install --upgrade pip
COPY requirements.txt /tmp/
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
RUN pip install --requirement /tmp/requirements.txt

# Start the container with a bash shell
CMD ["/bin/bash"]