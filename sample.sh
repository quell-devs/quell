#!/bin/bash
# sample script to generate sample data, train model, run validation, and run inference

# generate sample data in ./sample
poetry run quell-sample-data
poetry run quell-preview-data

# set parameters
BATCH=2
LR=0.001
EPOCHS=40
SIZE_I=32
SIZE_J=32
SIZE_K=32
RUN_NAME=unet_sample

# train model
echo "Training Quell denoising model with name: ${RUN_NAME}"
poetry run quell train \
    --csv sample/sample.csv \
    --preprocessed-dir sample/ \
    --batch-size $BATCH \
    --learning-rate $LR \
    --epochs $EPOCHS \
    --size-i ${SIZE_I} \
    --size-j ${SIZE_J} \
    --size-k ${SIZE_K} \
    --output-dir outputs/$RUN_NAME \
    --run-name $RUN_NAME \
    --cache-memory \
    --project-name quell-sample \
    --wandb

# run validation
echo "Running baseline validation - no denoising"
poetry run quell-identity validate \
    --csv sample/sample.csv \
    --preprocessed-dir sample/ \
    --batch-size $BATCH \
    --size-i ${SIZE_I} \
    --size-j ${SIZE_J} \
    --size-k ${SIZE_K} \
    --cache-memory \
    --result-csv validation_sample.csv \
    --eval-name baseline

echo "Running Quell validation - with denoising"
poetry run quell validate \
    --pretrained outputs/$RUN_NAME/export.pkl \
    --csv sample/sample.csv \
    --preprocessed-dir sample/ \
    --batch-size $BATCH \
    --size-i ${SIZE_I} \
    --size-j ${SIZE_J} \
    --size-k ${SIZE_K} \
    --cache-memory \
    --result-csv validation_sample.csv \
    --eval-name ${RUN_NAME}

# run inference on 'thin slices'
poetry run quell infer \
        --pretrained outputs/$RUN_NAME/export.pkl \
        --item sample/noisy_2.nrrd \
        --output sample/denoised_2.nrrd \
        --size-i ${SIZE_I} \
        --size-j ${SIZE_J} \
        --size-k ${SIZE_K} \
        --overlap 16

# generate preview of inference
poetry run quell-preview-data
