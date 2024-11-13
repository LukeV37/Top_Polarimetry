#/bin/bash
cd ../pythia-v8.312
./configure --with-root --with-hepmc2=../hepmc-v2.06.11/hepmc-install --prefix=$PWD --with-gzip
make -j4
