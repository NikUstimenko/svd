import numpy as np
import acoustotreams
import treams
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

lmax = 2
radius = 25e-6
c0 = 1500
rho0 = 998
c1 = 600
rho1 = 360
n = c0 / 343
water = acoustotreams.AcousticMaterial(rho0, c0)
materials = [acoustotreams.AcousticMaterial(rho1, c1), water]
kpar = np.array([0, 0])
wavelengths = np.linspace(105e-6, 106e-6, 1001)
freq = c0 / wavelengths
k0 = 2 * np.pi * n / wavelengths
L = np.linspace(66e-6, 72e-6, 1001)

def compute_svd(i, j):
    sphere = acoustotreams.AcousticTMatrix.sphere(lmax, k0[j], radius, materials)
    lattice = treams.Lattice.square(L[i])
    Teff = sphere.latticeinteraction.solve(lattice, kpar / n)
    sigma = np.linalg.svd(np.array(Teff), compute_uv=False)
    pwb = acoustotreams.ScalarPlaneWaveBasisByComp.diffr_orders(kpar / n, lattice, 1.01 * 2 * np.pi / L[i])
    SMat = acoustotreams.AcousticSMatrices.from_array(Teff, pwb)
    array = np.array(SMat)
    ng = int(np.sqrt(np.size(array[0, 0, :, :])))
    S = np.zeros((2*ng, 2*ng), complex)
    S[0:ng, 0:ng] = array[0, 0, :, :]
    S[ng:2*ng, 0:ng] = array[0, 1, :, :]
    S[0:ng, ng:2*ng] = array[1, 0, :, :]
    S[ng:2*ng, ng:2*ng] = array[1, 1, :, :]
    xi = np.linalg.svd(S, compute_uv=False)
    plw = acoustotreams.plane_wave_scalar(kpar, k0=k0[j], basis=pwb, material=water, modetype="down")
    return j, i, SMat.tr(plw)[0], 1/sigma[0], 1/xi[0]

results = Parallel(n_jobs=-1)(delayed(compute_svd)(i, j) for i in range(len(L)) for j in range(len(wavelengths)))
refl = np.zeros((len(wavelengths), len(L)))
sigma1 = np.zeros((len(wavelengths), len(L)))
xi1 = np.zeros((len(wavelengths), len(L)))

for j, i, result1, result2, result3 in results:
    refl[j, i] = result1
    sigma1[j, i] = result2
    xi1[j, i] = result3

L, Wavelengths = np.meshgrid(L, wavelengths)
mpl.rcParams['pdf.fonttype'] = 42 
fs = 12

fig, ax = plt.subplots(3, 1)
mesh = ax[0].pcolormesh(L * 1e6, Wavelengths * 1e6, refl, cmap='YlOrBr', vmin=0, vmax=1, rasterized=True) 
cbar = fig.colorbar(mesh, ax=ax[0])
ax[0].set_ylabel("Wavelength ($\\mu$m)", fontsize=fs)
cbar.set_label('Reflectance', fontsize=fs)
ax[0].tick_params(axis='x', bottom=False, labelbottom=False)
ax[0].tick_params(labelsize=fs)
cbar.ax.tick_params(labelsize=fs)

mesh = ax[1].pcolormesh(L * 1e6, Wavelengths * 1e6, sigma1, cmap='Blues', norm=LogNorm(), rasterized=True) 
cbar = fig.colorbar(mesh, ax=ax[1])
ax[1].set_ylabel("Wavelength ($\\mu$m)", fontsize=fs)
cbar.set_label('$\\sigma_1^{-1}$', fontsize=fs)
ax[1].tick_params(axis='x', bottom=False, labelbottom=False)
ax[1].tick_params(labelsize=fs)
cbar.ax.tick_params(labelsize=fs)
 
mesh = ax[2].pcolormesh(L * 1e6, Wavelengths * 1e6, xi1, cmap='Greens', norm=LogNorm(), rasterized=True) 
cbar = fig.colorbar(mesh, ax=ax[2])
ax[2].set_ylabel("Wavelength ($\\mu$m)", fontsize=fs)
cbar.set_label('$\\xi_1^{-1}$', fontsize=fs)
ax[2].set_xlabel("L ($\\mu$m)", fontsize=fs)
ax[2].tick_params(labelsize=fs)
cbar.ax.tick_params(labelsize=fs)
plt.show()