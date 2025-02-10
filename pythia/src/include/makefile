SUBMODULES_PATH=../../../submodules/

CPP=g++
LD=g++
SPECIALFLAGS=-O2
ROOTCFLAGS=$(shell root-config --cflags)
ROOTLIBS=$(shell root-config --libs)

CFLAGS = $(SPECIALFLAGS) -I.
LFLAGS = -L.

RCXX=$(CFLAGS) $(ROOTCFLAGS)
RLXX=$(LFLAGS) $(ROOTLIBS)

test_dataset: test_dataset.cpp
	@g++ test_dataset.cpp -o test_dataset -w  -I $(SUBMODULES_PATH)/pythia-v8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread  -L $(SUBMODULES_PATH)/pythia-v8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/pythia-v8.312/lib -lpythia8 -ldl $(shell root-config --libs --cflags)

clean:
	@rm -f test_dataset
	@rm -f histo.root
