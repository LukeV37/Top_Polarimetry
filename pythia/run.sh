#!/bin/bash

if [ -z "$1" ]; then
    echo "Must enter 1 agruments"
    echo "1: Dataset Tag (from MadGraph)"
    exit 1
fi

cd src
make generate_dataset
./run_dataset $1
make clean
cd ..
