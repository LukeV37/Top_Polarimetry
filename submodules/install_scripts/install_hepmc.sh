#!/bin/bash
cd ../hepmc-v2.06.11
mkdir hepmc-build
mkdir hepmc-install
cd hepmc-build
cmake -DCMAKE_INSTALL_PREFIX=../hepmc-install \
      -Dmomentum:STRING=MEV \
      -Dlength:STRING=MM \
      ../
make -j8
make install
