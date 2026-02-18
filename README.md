This repository provides a set of scripts to perform the singular value decomposition of the effective T-matrix and the S-matrix of various structures. In particular, they can be used to investigate bound states in the continuum in metasurface-based systems. The following systems are covered:

* [x] Periodic metasurface of dielectric spheres
* [x] Periodic metasurface on a substrate
* [x] Finite array
* [x] Double-layer metasurface of dielectric rods

To use these codes, the version of Python should be 3.10 or 3.11 
and you have to install `numpy` and `scipy<1.17` as well as `treams>0.4` 
```sh
pip install treams
```
For acoustic systems, also install `acoustotreams>0.2.5`
```sh
pip install acoustotreams
```

When using this code please cite:

[N. Ustimenko, I. Fernandez-Corbaton, and C. Rockstuhl, Singular value decomposition to describe bound states in the continuum in periodic metasurfaces, arXiv 2602.15741 (2026).](https://arxiv.org/abs/2602.15741)