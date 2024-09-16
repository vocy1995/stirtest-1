DATASETLOCATION=$1
OUTPUTLOCATION=$2 # where the written data will be put

docker run -it --rm --gpus "device=0" --net host --runtime=nvidia --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 -v /var/run/docker.sock:/var/run/docker.sock --mount src=$DATASETLOCATION,target=/workspace/data,type=bind --mount src=`pwd`,target=/workspace/stir,type=bind --mount src=$OUTPUTLOCATION,target=/workspace/STIRMetrics-main/output,type=bind -w /workspace/stir stircontainer:2.1 python src/dataset/flow2d.py