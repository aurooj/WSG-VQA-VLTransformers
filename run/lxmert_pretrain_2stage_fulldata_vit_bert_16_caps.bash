# The name of experiment
name=16caps_itm_mlm_qa_552_grid_x_allqa_vit_bert_stage_1

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
    --skipConnection --crossAttnType no_cross\
    --batchSize 1024 --optim bert --lr 1e-4 --epochs 20 \
    --NUM_PRIM_CAPS 16 --NUM_VIS_CAPS 16 \
    --vitInit --startIndex 7 \
    --tqdm --output $output ${@:2}



# The name of experiment
name=16caps_itm_mlm_qa_552_grid_x_allqa_vit_bert_stage_2

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
    --llayers 5  --rlayers 5 --xlayers 2 \
   --loadLXMERT snap/pretrain/16caps_itm_mlm_qa_552_grid_x_allqa_vit_bert_stage_1/BEST_EVAL_LOSS \
    --skipConnection --crossAttn --crossAttnType cross --freezeWeights\
    --batchSize 1024 --optim bert --lr 1e-4 --epochs 15 \
    --NUM_PRIM_CAPS 16 --NUM_VIS_CAPS 16 \
    --vitInit --startIndex 7 \
    --tqdm --output $output ${@:2}


