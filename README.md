## Quick Start
Clone the repo over ssh using:
```
git clone --recursive git@github.com:LukeV37/Pileup_Project.git
```

Or clone the repo over https using:
```
git clone --recursive https://github.com/LukeV37/Pileup_Project.git
```

Install the submodules:
```
./build_submodules.sh
```

Please be patient while submodules build...

## How To Generate Datasets

### Madgraph
To run madgraph simulation, run the following
```
cd madgraph
./run.sh
```
The lhe file will generated in the `pp_tt_semi_full/Events/run_01/` dir. 

### Pythia
To shower in pythia, run the following
```
cd pythia
./run.sh
```
The hepmc file will generated in the `output/` dir. 


## Dependencies
Runs on Ubuntu 22.04 and wsl2

Required Dependencies:
<ul>
  <li>gzip</li>
</ul>
