# The name of experiment
name=lxmert_pretrain_itm_mlm_qa_552_grid_x_self_allqa_no_init_stage_1

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
    --fromScratch --skipConnection --crossAttnType no_cross\
    --batchSize 1024 --optim bert --lr 1e-4 --epochs 20 \
     --tqdm --output $output ${@:2}



# The name of experiment
name=lxmert_pretrain_itm_mlm_qa_552_grid_x_self_allqa_no_init_stage_2

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training
#batch size reduced due to additional cross attn layer (large model)
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/lxmert_pretrain.py \
     --taskMatched --taskMaskLM --taskQA \
    --train mscoco_train,vgnococo --valid mscoco_minival \
    --llayers 5  --rlayers 5 --xlayers 2\
   --loadLXMERT snap/pretrain/lxmert_pretrain_itm_mlm_qa_552_grid_x_self_allqa_no_init_stage_1/BEST_EVAL_LOSS \
    --skipConnection --crossAttn --crossAttnType self --freezeWeights\
    --batchSize 1024 --optim bert --lr 1e-4 --epochs 15 \
    --tqdm --output $output ${@:2}


