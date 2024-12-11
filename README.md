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
2) Setup conda environment
```
# in FiGS
conda env create -f environment_x86.yml
conda activate figs
```
3) Download example gsplat data
```
# in FiGS
gdown --folder https://drive.google.com/drive/folders/1YdezcMckg2INXGC33JA3CNbHgjXzfTzY?usp=drive_link
```
