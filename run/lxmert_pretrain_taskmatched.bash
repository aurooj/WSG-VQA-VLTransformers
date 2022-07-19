# The name of experiment
name=lxmert_pretrain_ITM_MLM_QA_55_MSCOCO_VG_correct_im2id

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/lxmert_pretrain.py \
     --taskMatched --taskMaskLM --taskQA \
    --train mscoco_train,vgnococo --valid mscoco_minival \
    --llayers 5  --rlayers 5 --xlayers 0\
    --fromScratch --skipConnection --excludeSet gqa\
    --batchSize 256 --optim bert --lr 1e-4 --epochs 20 \
    --tqdm --output $output ${@:2}

