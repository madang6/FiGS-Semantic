# FiGS
Installation steps
1) Install acados
```
cd acados
git submodule update --recursive --init

mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4

# Add acados paths to bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/admin/StanfordMSL/FiGS/acados/lib"
export ACADOS_SOURCE_DIR="/home/admin/StanfordMSL/FiGS/acados"
```
