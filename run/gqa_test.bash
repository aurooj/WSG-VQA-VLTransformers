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
     --train train --valid "" \
    --llayers 5 --xlayers 2  --rlayers 5 --outputAttn --skipConnection \
    --tqdm --output $output ${@:3}
