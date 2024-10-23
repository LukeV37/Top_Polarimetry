SUBMODULES_PATH=../../submodules/

generate_hepmc: generate_hepmc.cpp
	@g++ generate_hepmc.cpp -o run_hepmc -w  -I $(SUBMODULES_PATH)/pythia-v8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread  -L $(SUBMODULES_PATH)/pythia-v8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/pythia-v8.312/lib -lpythia8 -ldl -I$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/include -L$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/hepmc-v2.06.11/hepmc-install/lib -lHepMC -DHEPMC2

clean:
	@rm -f run_hepmc
