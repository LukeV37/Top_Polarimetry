SUBMODULES_PATH=../../submodules/

generate_dataset: generate_dataset.cpp
	@g++ generate_dataset.cpp -o run_dataset -w  -I $(SUBMODULES_PATH)/pythia-v8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread  -L $(SUBMODULES_PATH)/pythia-v8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/pythia-v8.312/lib -lpythia8 -ldl -I$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/include -L$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -lHepMC -DHEPMC2 $(shell root-config --libs --cflags)

clean:
	@rm -f run_dataset
