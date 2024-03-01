#FROM  nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

# All users can use /home/user as their home directory
ENV HOME=/home/
RUN chmod 777 /home/
ARG DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get -y update && \
    apt-get -y install \
        python3 \
        python3-dev \
        python3-pip \
        wget \
        sudo \
        gfortran \
        pcre2-utils \
        libxt-dev \
        libxml2-dev \
        libssl-dev \
        libfontconfig1-dev \
        libcurl4-gnutls-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        curl \
        muon \
        nano \
        build-essential \
        ca-certificates \
        cargo \
        cmake \
        default-jdk \
        fonts-roboto \
        htop \
        libcairo2-dev \
        libclang-dev \
        libfreetype6-dev \
        libgdal-dev \
        libgeos-dev \
        libjpeg-dev \
        libproj-dev \
        libpng-dev \
        libpq-dev \
        libsodium-dev \
        libtiff5-dev \
        libudunits2-dev \
        libx11-dev \
        openjdk-8-jdk \
        openjdk-8-jre \
        pandoc \
        ttf-mscorefonts-installer \
        xorg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh && \
    conda clean -afy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Create a Python 3.10.0 environment
RUN conda create -y --name py310 python=3.10.0 &&\
    conda clean -ya
ENV CONDA_DEFAULT_ENV=py310
ENV CONDA_PREFIX=/usr/local/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Installing CUDA
#RUN conda install -c conda-forge cudatoolkit=11.8.0
#RUN python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
#RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d
#ENV CUDNN_PATH=/usr/local/envs/py310/lib/python3.10/site-packages/nvidia/cudnn
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
#RUN echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#RUN . source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#Installing R.4.2.2
ARG R_VERSION=4.2.2
ARG OS_IDENTIFIER=ubuntu-1804
RUN wget https://cdn.posit.co/r/${OS_IDENTIFIER}/pkgs/r-${R_VERSION}_1_amd64.deb && \
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -f -y ./r-${R_VERSION}_1_amd64.deb && \
    ln -s /opt/R/${R_VERSION}/bin/R /usr/bin/R && \
    ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/bin/Rscript && \
    ln -s /opt/R/${R_VERSION}/lib/R /usr/lib/R && \
    rm r-${R_VERSION}_1_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

RUN chown root:root /usr/bin/sudo && chmod 4755 /usr/bin/sudo
RUN R -e "install.packages('devtools',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('clue',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('bootcluster',dependencies=TRUE, repos='http://cran.rstudio.com/')"
#RUN R -e 'remotes::install_github("cran/OTclust", dependencies = TRUE, force = TRUE, upgrade = TRUE, lib = Sys.getenv("R_LIBS_USER"))'
RUN R -e 'remotes::install_github("cran/OTclust", dependencies = TRUE, force = TRUE, upgrade = TRUE)'
RUN R -e "install.packages('tsne',dependencies=TRUE, repos='http://cran.rstudio.com/')"

COPY requirements.txt .
RUN ["/bin/bash", "-c", "pip install -r <(cat requirements.txt)"]

ADD install_environment.sh install_environment.sh
RUN bash install_environment.sh