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

SRC1=generate_dataset.cpp generate_dataset_dict.cpp
SRC2=selections.cxx

generate_dataset_dict.cpp: ./include/LinkDef.h
	@rootcling generate_dataset_dict.cpp ./include/LinkDef.h

%.o: %.cpp
	@$(CPP) $(RCXX) -c $< -w -I $(SUBMODULES_PATH)/pythia-v8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SUBMODULES_PATH)/pythia-v8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/pythia-v8.312/lib -lpythia8 -ldl -I$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/include -L$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -lfastjet -I$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/include -L$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -lHepMC -DHEPMC2

generate_dataset: $(SRC1:.cpp=.o)
	@$(LD) $(SRC1:.cpp=.o) $(RLXX) -o run_dataset -w  -I $(SUBMODULES_PATH)/pythia-v8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SUBMODULES_PATH)/pythia-v8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/pythia-v8.312/lib -lpythia8 -ldl -I$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/include -L$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -lfastjet -I$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/include -L$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -lHepMC -DHEPMC2 $(shell root-config --libs --cflags)

selections_dict.cpp: ./include/LinkDef.h
	@rootcling selections_dict.cpp ./include/LinkDef.h

selections: $(SRC2:.cpp=.o)
	@$(LD) $(SRC2:.cpp=.o) $(RLXX) -o run_selections -w -I$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/include -L$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/fastjet-v3.4.2/fastjet-install/lib -lfastjet $(shell root-config --libs --cflags)

clean:
	@rm -f run_dataset
	@rm -f generate_*_dict* *.o
	@rm -f run_selections
	@rm -f selections*_dict* *.o

