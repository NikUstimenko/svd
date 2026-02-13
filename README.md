This repository provides a set of scripts to perform the singular value decomposition of the effective T-matrix and the S-matrix of various structures. In particular, they can be used to investigate bound states in the continuum in metasurface-based systems. The following systems are covered:

* [x] Periodic metasurface of dielectric spheres
* [x] Periodic metasurface on a substrate
* [x] Finite array
* [x] Double-layer metasurface of dielectric rods

To use these codes, the version of Python must be 3.10 or 3.11 
and you have to install `numpy` and `scipy<1.17` as well as `treams>0.4` 
```sh
pip install treams
```
For acoustic systems, also install `acoustotreams`
```sh
pip install acoustotreams
```

When using this code please cite: