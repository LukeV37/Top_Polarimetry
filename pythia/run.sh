#!/bin/bash

if [ -z "$3" ]; then
    echo "Must enter 3 agruments"
    echo -e "\t1: Dataset Tag (from MadGraph)"
    echo -e "\t2: Number of runs (from MadGraph)"
    echo -e "\t3: Max num cpu cores"
    exit 1
fi

tag=$1
runs=$2
max_cpu_cores=$3

mkdir -p "WS_${tag}"
mkdir -p "WS_${tag}/logs"

cd src
make generate_dataset

echo "Please be patient while Pythia performs hadronization..."

job=0
batch=1
for (( i=0 ; i<$runs ; i++ ));
do
    echo -e "\t\tSubmitting job to shower run $i"
    ./run_dataset $tag $i > "../WS_${tag}/logs/dataset_${tag}_${i}.log" 2>&1 &
    job=$((job+1))
    if [ $job == $max_cpu_cores ]; then
        echo -e "\tStopping jobs submissions! Please wait for batch $batch to finish..."
        wait
        job=0
        batch=$((batch+1))
    fi
done
wait
make clean
cd ..

echo -e "\tPythia Showering Done!"
