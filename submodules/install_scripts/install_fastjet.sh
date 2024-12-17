#!/bin/bash
cd ../fastjet-v3.4.2
mkdir fastjet-install
# run autogen twice to prevent ltmain.sh "not found"
# first time it uses ../.. for libtoolize
# second time it uses . for libtoolize
# I cannot figure out the problem other than running ./autogen.sh twice
./autogen.sh --prefix="$PWD/fastjet-install/"
./autogen.sh --prefix="$PWD/fastjet-install/"
make -j8
make install
rm -f ../../ltmain.sh
