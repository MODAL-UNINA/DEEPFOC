#!/usr/bin/env bash
set -ex

export PATH="/usr/bin:${PATH}"

which ld        # → /usr/bin/ld
which gcc       # → /usr/bin/gcc
which gfortran  # → /usr/bin/gfortran

ls 
cd fpfit

make clean

make fpplot CC=/usr/bin/gcc FC=/usr/bin/gfortran
make fpfit 

cp fpfit ../fpyfit/
cp fpplot ../fpyfit/
make clean

cd ..

chmod +x ./fpyfit/fpfit
chmod +x ./fpyfit/fpplot

python -uB build_fortran_modules.py
