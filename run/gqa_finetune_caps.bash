# The name of this experiment.
name=$2

# Save logs and models under snap/gqa; make backup.
output=snap/gqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa.py \
    --train train --valid valid \
    --llayers 5 --xlayers 2 --rlayers 5 --NUM_PRIM_CAPS 32 --NUM_VIS_CAPS 32 --skipConnection --crossAttn\
    --loadLXMERT snap/pretrain/mm_capsules_pretrain_552_stage2_fixed_continued/BEST_EVAL_LOSS \
    --batchSize 32 --optim bert --lr 1e-5 --epochs 20\
    --tqdm --output $output ${@:3}
