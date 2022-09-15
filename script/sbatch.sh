#!/bin/bash

if [[ $# -eq 0 ]]; then
    echo "./sbatch.sh [CONFIG.json]"
    echo "            (--name NAME)"
    echo "            (--ckpt CKPT_FN)"
    echo "            (--epoch EPOCH)"
    echo "            (--cg {CalphaBasedModel, ResidueBasedModel})"
    echo "            (--gpu N_GPU, default=1)"
    echo "            (--cpu N_CPU, default=8)"
    exit -1
fi
config=$1
name=$(basename $config .json)

n_gpu=1
n_cpu=8
args=""
#
OPTS=$(getopt --alternative --longoptions cpu:,gpu:,name:,ckpt:,epoch:,cg -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in 

    --gpu )
        n_gpu=$2
        shift 2
        ;;

    --cpu )
        n_cpu=$2
        shift 2
        ;;

    -- )
        shift;
        break
        ;;

    * )
        args="$args $1 $2"
        shift 2
        ;;

    esac
done

cmd="./script/train.py --config $config $args --requeue"

echo $cmd
echo $cmd \
    | sbatch.script --name $name \
    --output logs/$name.log \
    --partition feig \
    --conda dgl \
    --gpu $n_gpu \
    --cpu $n_cpu
