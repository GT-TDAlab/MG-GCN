# MG-GCN

MG-GCN: multi-GPU GCN training framework.

After cloning our repository, run `git submodule update --init` to download the submodules.

Our software depends on a recent CUDA installation, tested on CUDA 11.4

For parallel preprocessing, our software makes use of the parallel standard library, GCC implementation of the standard library depends on TBB.
A recent version of TBB is required, the following is the most recent TBB version that is compatible for our purpose: [tbb release](https://github.com/oneapi-src/oneTBB/archive/v2020.2.zip)

One can use the following command to compile and install TBB:

`python3 build/build.py --prefix="<PATH-TO-INSTALL>" --install-libs --install-devel`

If tbb is not found, add environment variable: `export TBB_ROOT=<PATH-TO-INSTALL>`

A recent version of NCCL is also required. You can follow the instructions here to compile and install it: [nccl github](https://github.com/NVIDIA/nccl)

GCC version 9 or above is required to compile our software.

When all prerequisites are installed, one can create a build directory and compile our software as follows:
```
mkdir build
cd build
cmake ..
make -j
```

To download and preprocess the datasets used in our experiments, first change directory into `test/data`. Then run `prep.py` as follows:
```
cd test/data
mkdir permuted
python3 prep.py -s=0
python3 prep.py -s=1
```

These commands will download the reddit dataset and output them into the test/data directory. If you want to download other datasets, uncomment the corresponding lines at the end of `prep.py` and run our script as above. Note that this script requires an installation of dgl, ogb and some other python packages.

Finally, to run our code on the reddit dataset, use the following line from the root directory of our repository:
```
build/src/mg_gcn -P 4 -R 1 train test/data/permuted/reddit/ 3 128 128 128
```

`-P` is for the number of GPUS, 3 128 128 128 denotes the number of hidden layers and their dimensions.
