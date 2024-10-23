#!/bin/bash
cd src
make generate_hepmc
./run_hepmc
make clean
cd ..
