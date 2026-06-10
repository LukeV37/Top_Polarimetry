#!/bin/bash
cd ../zlib-1.3.2
mkdir zlib-install
./configure --prefix="$PWD/zlib-install"
make -j8
make install
