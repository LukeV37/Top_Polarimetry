SUBMODULES_PATH=../../submodules/

CPP=g++
LD=g++
SPECIALFLAGS=-O2
ROOTCFLAGS=$(shell root-config --cflags)
ROOTLIBS=$(shell root-config --libs)

CFLAGS = $(SPECIALFLAGS) -I.
LFLAGS = -L.

RCXX=$(CFLAGS) $(ROOTCFLAGS)
RLXX=$(LFLAGS) $(ROOTLIBS)

SRC=generate_dataset.cpp generate_dataset_dict.cpp

generate_dataset_dict.cpp: ./include/LinkDef.h
	@rootcling generate_dataset_dict.cpp ./include/LinkDef.h

%.o: %.cpp
	@$(CPP) $(RCXX) -c $< -w -I $(SUBMODULES_PATH)/pythia-v8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SUBMODULES_PATH)/pythia-v8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/pythia-v8.312/lib -lpythia8 -ldl -I$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/include -L$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -lfastjet -I$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/include -L$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -lHepMC -DHEPMC2

generate_dataset: $(SRC:.cpp=.o)
	@$(LD) $(SRC:.cpp=.o) $(RLXX) -o run_dataset -w  -I $(SUBMODULES_PATH)/pythia-v8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SUBMODULES_PATH)/pythia-v8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/pythia-v8.312/lib -lpythia8 -ldl -I$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/include -L$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -lfastjet -I$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/include -L$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -lHepMC -DHEPMC2 $(shell root-config --libs --cflags)

test_dataset: test_dataset.cpp
	@g++ test_dataset.cpp -o test_dataset -w  -I $(SUBMODULES_PATH)/pythia-v8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread  -L $(SUBMODULES_PATH)/pythia-v8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/pythia-v8.312/lib -lpythia8 -ldl $(shell root-config --libs --cflags)


clean:
	@rm -f run_dataset
	@rm -f generate_*_dict* *.o
