# Based on https://stackoverflow.com/a/65495386

# Use nvidia/cuda image
FROM nvidia/cuda:10.2-devel-ubuntu18.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
RUN conda update conda \
    && conda env create --name cut python=3.8.10
RUN echo "conda activate cut" >> ~/.bashrc
ENV PATH /opt/conda/envs/cut/bin:$PATH
ENV CONDA_DEFAULT_ENV $cut

# install dependencies
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
WORKDIR /3D-MRI-style-transfer
# COPY . .
RUN pip install -r requirements.txt

RUN echo "Successfully build image!"