FROM docker.io/nvidia/cuda:11.4.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y

# Dex dependencies
RUN apt-get install -y llvm-9-dev clang-9 libpng-dev
RUN apt-get install -y git
RUN apt-get install -y curl
RUN apt-get install -y netbase
RUN apt-get install -y pkg-config
RUN bash -c 'curl -sSL https://get.haskellstack.org/ | sh'

# Dex
RUN git clone https://github.com/google-research/dex-lang.git
# TODO Check out specific commit???
# TODO Happened to get c19d9a3d948de6f6d0b2b5137b4ae091433fd993
RUN cd dex-lang && make
ENV LD_LIBRARY_PATH /usr/local/cuda/compat 
RUN cd dex-lang && PREFIX=/usr/local/bin make install

# Nx dependencies
# IMPORTANT -> I failed to make mix happy with the bound volume,
# IMPORTANT -> but I also haven't deleted this setup here.
RUN curl -O https://packages.erlang-solutions.com/erlang-solutions_2.0_all.deb
RUN dpkg -i erlang-solutions_2.0_all.deb
RUN apt-get update
# TODO Control versions here?
RUN apt-get install -y esl-erlang
RUN apt-get install -y elixir
ENV EXLA_TARGET cuda
ENV XLA_TARGET cuda111

# Locales, which iex otherwise complains about
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Python
RUN apt -y install python3 python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Jax
RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install sklearn torch

# Editor
RUN apt -y install vim

# User
RUN useradd -rm -u 1001 -g root fundiff
WORKDIR /home/fundiff
USER fundiff
