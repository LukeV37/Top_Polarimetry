#!/bin/bash
cd src
make generate_dataset
./run_dataset
make clean
cd ..
