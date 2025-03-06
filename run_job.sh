#!/bin/bash

# Read options from config
source job.config

# Get current directory
WORKING_DIR=$(pwd)

# Setup python venv
source setup.sh

# Run MadGraph Generation
if [ "$bypass_madgraph" = false ]; then
  start=`date +%s`
  cd madgraph
  ./run.sh $tag1 $process1 $num_runs $num_events_per_run
  ./run.sh $tag2 $process2 $num_runs $num_events_per_run
  cd $WORKING_DIR
  end=`date +%s`
  runtime=$((end-start))
  echo -e "\tTime (sec): $runtime"
fi

# Run Pythia Showering
if [ "$bypass_pythia" = false ]; then
  start=`date +%s`
  cd pythia
  ./run.sh $tag1 $num_runs
  ./run.sh $tag2 $num_runs
  cd $WORKING_DIR
  end=`date +%s`
  runtime=$((end-start))
  echo -e "\tTime (sec): $runtime"
fi

# Run preprocessing to match fat jets to top quarks
if [ "$bypass_preprocessing" = false ]; then
  echo "Please be patient for Preprocessing..."
  start=`date +%s`
  cd model
  mkdir -p "${dir_preprocessing1}"
  mkdir -p "${dir_preprocessing2}"
  mkdir -p "${dir_datasets}"
  for (( i=0 ; i<$num_runs ; i++ ));
  do
    mkdir -p "${dir_preprocessing1}/run_$i"
    python -u Preprocessing.py $tag1 $i $dir_preprocessing1 $dir_datasets > "${dir_preprocessing1}/run_$i/preprocessing.log" &
  done
  wait
  for (( i=0 ; i<$num_runs ; i++ ));
  do
    mkdir -p "${dir_preprocessing2}/run_$i"
    python -u Preprocessing.py $tag2 $i $dir_preprocessing2 $dir_datasets > "${dir_preprocessing2}/run_$i/preprocessing.log" &
  done
  wait
  cd $WORKING_DIR
  end=`date +%s`
  runtime=$((end-start))
  echo -e "\tPreprocessing Done!"
  echo -e "\tTime (sec): $runtime"
fi

# Run batching to pad dataset
if [ "$bypass_batch" = false ]; then
  echo "Please be patient for Batching..."
  start=`date +%s`
  cd model
  for (( i=0 ; i<$num_runs ; i++ ));
  do
    mkdir -p "${dir_datasets}/logs"
    python -u Batch_BCE.py $tag1 $tag2 $i $dir_datasets > "${dir_datasets}/logs/batch.log" &
  done
  wait
  python -u Combine_Batches.py $tag1 $tag2 $num_runs $dir_datasets
  cd $WORKING_DIR
  end=`date +%s`
  runtime=$((end-start))
  echo -e "\tBatching Done!"
  echo -e "\tTime (sec): $runtime"
fi

# Run training script
if [ "$bypass_train" = false ]; then
  echo "Please be patient for Training..."
  start=`date +%s`
  cd model
  mkdir -p $dir_training
  python -u Jet_Attention_Model_BCE.py $tag1 $tag2 $epochs $step $dir_datasets $dir_training | tee "training_${tag1}_${tag2}.log"
  cd $WORKING_DIR
  end=`date +%s`
  runtime=$((end-start))
  echo -e "\tTraining Done!"
  echo -e "\tTime (sec): $runtime"
fi
