#!/bin/bash
# Usage:
# ./experiments/scripts/rfcn_test_renms_acc.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/rfcn_test_renms_acc.sh 0 ResNet50 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_0712_trainval"
    TEST_IMDB="voc_0712_test"
    PT_DIR="pascal_voc"
    #ITERS=10000
     ITERS=50000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_val"
    PT_DIR="coco"
    ITERS=1920000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/rfcn_test_renms_acc_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


time ./tools/test_net_renms.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/rfcn_end2end/test_agnostic.prototxt \
  --net /home/xyt/py-R-FCN-Bronze-Anchors/output_end2end/rfcn_end2end_ohem/voc_0712_trainval/resnet50_rfcn_ohem_iter_70000.caffemodel \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/rfcn_end2end_ohem.yml \
  ${EXTRA_ARGS}
