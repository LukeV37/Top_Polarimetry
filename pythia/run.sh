#!/bin/bash

if [ -z "$6" ]; then
    echo "Must enter 4 agruments"
    echo -e "\t1: Dataset Tag (from MadGraph)"
    echo -e "\t2: Dataset Tag (for Pythia)"
    echo -e "\t3: Number of runs (from MadGraph)"
    echo -e "\t4: Max num cpu cores"
    echo -e "\t5: R parameter for clustering"
    echo -e "\t6: Min Jet pT Cut in GeV"
    exit 1
fi

MG_tag=$1
PY_tag=$2
runs=$3
max_cpu_cores=$4
R=$5
minpT=$6

mkdir -p "WS_${PY_tag}"
mkdir -p "WS_${PY_tag}/logs"

cd src
make generate_dataset
make selections

echo "Please be patient while Pythia performs hadronization..."

job=0
batch=1
for (( i=0 ; i<$runs ; i++ ));
do
    echo -e "\t\tSubmitting job to shower run $i"
    #(./run_dataset $tag $i; ./run_selections $tag $i) > "../WS_${tag}/logs/dataset_${tag}_${i}.log" 2>&1 &
    ./run_dataset $MG_tag $PY_tag $i $R $minpT > "../WS_${PY_tag}/logs/dataset_${PY_tag}_${i}.log" 2>&1 &
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
