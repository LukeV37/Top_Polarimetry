#!/bin/bash
WORK_DIR=$(pwd)
if [ ! -f ./submodules/torch/bin/activate ]; then
    cd submodules
    python3 -m venv torch
    cd torch
    source ./bin/activate
    pip install --upgrade pip
    pip install -r pip_requirements.txt
else
    source ./submodules/torch/bin/activate
fi
cd $WORK_DIR
