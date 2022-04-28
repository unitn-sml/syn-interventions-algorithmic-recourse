#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# See this for the original https://edu.itp.phys.ethz.ch/hs12/programming_techniques/openmpi.pdf

HOME=""
CORES=8

mkdir -p $HOME/local/src
mkdir -p $HOME/opt/openmpi

cd $HOME/local/src
wget https://download.open-mpi.org/release/open-mpi/v2.1/openmpi-2.1.1.tar.gz
tar -xf openmpi-2.1.1.tar.gz

cd openmpi-2.1.1
configure --prefix=$HOME/opt/openmpi
make -j$CORES all
make install

cd
rm $HOME/local/src/openmpi-2.1.1.tar.gz
rm -r $HOME/local/src/openmpi-2.1.1

echo "[*] Update bashrc"
echo "export PATH=\$PATH:\$HOME/opt/openmpi/bin" >> $HOME/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/opt/openmpi/lib" >> $HOME/.bashrc