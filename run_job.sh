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
  ./run.sh $tag $process $num_events
  cd $WORKING_DIR
fi

# Run Pythia Showering
if [ "$bypass_pythia" = false ]; then
  cd pythia
  ./run.sh $tag
  cd $WORKING_DIR
fi

# Run preprocessing to match fat jets to top quarks
if [ "$bypass_preprocessing" = false ]; then
  cd model
  mkdir -p $dir_preprocessing
  mkdir -p $dir_datasets
  python -u Preprocessing.py $tag $dir_preprocessing $dir_datasets | tee "${dir_preprocessing}/preprocessing.log"
  cd $WORKING_DIR
fi

# Run batching to pad dataset
if [ "$bypass_batch" = false ]; then
  cd model
  mkdir -p $dir_datasets
  python -u Batch_MSE.py $tag $dir_datasets | tee "${dir_datasets}/batch.log"
  cd $WORKING_DIR
fi

# Run training script
if [ "$bypass_train" = false ]; then
  cd model
  mkdir -p $dir_training
  python -u Jet_Attention_Model_MSE.py $tag $epochs $step $dir_datasets $dir_training | tee "${dir_training}/training.log"
  cd $WORKING_DIR
fi

echo "./run_all.sh Done!"
