#/bin/bash
cd ../pythia-v8.312
./configure --with-root --with-hepmc2=../hepmc-v2.06.11/hepmc-install --with-fastjet3=../fastjet-v3.4.2/fastjet-install --prefix=$PWD --with-gzip --with-gzip-include=../zlib-1.3.2/zlib-install/include --with-gzip-lib=../zlib-1.3.2/zlib-install/lib
make -j8
