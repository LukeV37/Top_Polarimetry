#!/bin/bash

# Read options from config
source job.config

# Get current directory
WORKING_DIR=$(pwd)

# Setup python venv
source setup.sh

# Run MadGraph Generation
if [ "$bypass_madgraph" = false ]; then
  cd madgraph
  ./run.sh $tag1 $process1 $num_events
  ./run.sh $tag2 $process2 $num_events
  cd $WORKING_DIR
fi

# Run Pythia Showering
if [ "$bypass_pythia" = false ]; then
  cd pythia
  cd src
  make generate_dataset
  ./run_dataset $tag1 > "../pythia_$tag1.log" &
  ./run_dataset $tag2 > "../pythia_$tag2.log" &
  wait
  make clean
  cd $WORKING_DIR
fi

# Run preprocessing to match fat jets to top quarks
if [ "$bypass_preprocessing" = false ]; then
  cd model
  mkdir -p $dir_preprocessing1
  mkdir -p $dir_preprocessing2
  python -u Preprocessing.py $tag1 $dir_preprocessing1 | tee "${dir_preprocessing1}/preprocessing1.log" &
  python -u Preprocessing.py $tag2 $dir_preprocessing2 | tee "${dir_preprocessing2}/preprocessing1.log" &
  wait
  cd $WORKING_DIR
fi

# Run batching to pad dataset
if [ "$bypass_batch" = false ]; then
  cd model
  python -u Batch_BCE.py $tag1 $tag2 | tee "batch.log"
  cd $WORKING_DIR
fi

# Run training script
if [ "$bypass_train" = false ]; then
  cd model
  python -u Jet_Attention_Model_BCE.py $tag1 $tag2 $epochs $step | tee "training_${tag1}_${tag2}.log"
  cd $WORKING_DIR
fi

echo "./run_all.sh Done!"
