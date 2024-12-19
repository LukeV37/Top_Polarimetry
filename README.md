## Quick Start
Clone the repo:
```
git clone --recursive git@github.com:LukeV37/Pileup_Project.git
```

Ensure dependencies are met, and install the submodules:
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
./post_process.sh
```
The lhe file file is generated, and the training labels are written to root file. 

### Pythia
To shower in pythia, run the following
```
cd pythia
./run.sh
```
The training data will be written to a root file. 

## Dependencies
Runs on most linux environments. Developed on Ubuntu 22.04.

Required Dependencies:
<ul>
  <li>python3</li>
  <li>ROOTv6</li>
  <li>g++</li>
  <li>gfortran</li>
  <li>gzip</li>
  <li>automake</li>
  <li>libtool</li>
  <li>autoconf</li>
</ul>
