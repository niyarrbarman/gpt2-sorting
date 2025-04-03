#!/bin/bash

pip3 install -r requirements.txt
echo "Requirements installed."

python3 gen_data.sh
echo "Data generation completed."

python3 tokenize_data.py
echo "Data tokenization completed."

python3 train.py
echo "Training completed."
