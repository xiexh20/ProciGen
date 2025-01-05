#!/bin/bash
# Download data for demo
mkdir -p example
cd example

# BEHAVE object templates
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/objects.zip
unzip objects.zip -d behave
rm objects.zip

# BEHAVE parameters
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-30fps-params-v1.tar
mkdir -p behave/params
tar -xvf behave-30fps-params-v1.tar -C behave/params
rm behave-30fps-params-v1.tar

# MGN SMPLD registration parameters
wget https://datasets.d2.mpi-inf.mpg.de/cvpr24procigen/mgn-smpld.zip
unzip mgn-smpld.zip -d mgn-smpld
rm mgn-smpld.zip

# Download one example ProciGen sequence
wget https://datasets.d2.mpi-inf.mpg.de/cvpr24procigen/ProciGen-mini.zip
unzip ProciGen-mini.zip -d ProciGen
rm ProciGen-mini.zip

# Additional assets for synthesizing and rendering
wget https://datasets.d2.mpi-inf.mpg.de/cvpr24procigen/ProciGen-assets-demo.zip
unzip ProciGen-assets-demo.zip -d assets
rm ProciGen-assets-demo.zip

echo "Example assets data downloaded success!"