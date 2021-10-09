docker build -t fundiff .

docker run --rm -it --gpus all --net=host --volume /media:/media --volume /media/ifueko/Data/Code/fundiff:/home/fundiff --shm-size=512G --name=fundiff fundiff

