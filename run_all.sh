#!/bin/bash

# Activate the virtual environment
source ~/env/bin/activate

# Array of models to run
models=("efficientnet" "custom")

for model in "${models[@]}"
do
    echo "=========================================================="
    echo "         STARTING OPTIMIZATION RUN FOR: $model            "
    echo "=========================================================="
    
    python train.py --model $model
    
    echo ""
    echo "Finished $model. Moving to next..."
    echo ""
done

echo "=========================================================="
echo "         ALL MODELS HAVE BEEN SUCCESSFULLY OPTIMIZED!     "
echo "=========================================================="
