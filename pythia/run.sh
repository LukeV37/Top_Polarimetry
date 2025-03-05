#!/bin/bash

if [ -z "$2" ]; then
    echo "Must enter 2 agruments"
    echo -e "\t1: Dataset Tag (from MadGraph)"
    echo -e "\t2: Number of runs (from MadGraph)"
    exit 1
fi

tag=$1
runs=$2

mkdir "WS_${tag}"
mkdir "WS_${tag}/logs"

cd src
make generate_dataset

for (( i=0 ; i<$runs ; i++ ));
do
    ./run_dataset $tag $i > "../WS_${tag}/logs/dataset_${tag}_${i}.log" 2>&1 &
done
wait
make clean
cd ..
