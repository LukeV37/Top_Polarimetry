#/bin/bash
cd ../pythia-v8.312
./configure --with-root --with-hepmc2=../hepmc-v2.06.11/hepmc-install --with-fastjet3=../fastjet-v3.4.2/fastjet-install --prefix=$PWD --with-gzip
make -j8
