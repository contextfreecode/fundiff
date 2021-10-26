# Setup
Assumes nvidia drivers installed. With Ubuntu 20.04, Use `sudo ubuntu-drivers autoinstall` and reboot for quick setup.

## Install docker and nvidia docker
`sudo apt install -y docker nvidia-docker2`

## Run docker build and execution script
`sudo ./run_docker.sh`

Note: the run docker script mounts `/home/fundiff` to the root location of the codebase on Ifueko's personal filesystem, which is mounted at `/media/ifueko/Data/Code/fundiff/`. Before running, change the root directory to the full filepath of the repository on your computer.
