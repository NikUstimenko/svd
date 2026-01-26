import numpy as np
import treams
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
treams.config.POLTYPE = "parity"

lmax = 1
radius = 200
materials = [treams.Material(3.5 * 3.5, 1, 0), treams.Material(1, 1, 0)]
nsub = 1.45
material_sub = treams.Material(nsub**2, 1, 0)
L = 1037.77895
lattice = treams.Lattice.square(L)
wavelengths = np.linspace(1482, 1490, 401) + 0.012
k0 = 2 * np.pi / wavelengths
theta = np.linspace(-5, 5, 401)

def compute_svd(i, j):
    sphere = treams.TMatrix.sphere(lmax, k0[j], radius, materials, poltype="parity")
    kpar = np.array([np.sin(theta[i] / 180 * np.pi) * k0[j], 0])
    Teff = sphere.latticeinteraction.solve(lattice, kpar)
    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 4.01 * 2 * np.pi / L)
    interface = treams.SMatrices.interface(pwb, k0[j], materials=[material_sub, 1], poltype="parity")
    propagation = treams.SMatrices.propagation([0, 0, radius], pwb, k0[j], 1, poltype="parity")
    array = treams.SMatrices.from_array(Teff, pwb)
    SMat = treams.SMatrices.stack([interface, propagation, array])
    array = np.array(SMat)
    ng = int(np.sqrt(np.size(array[0, 0, :, :])))
    S = np.zeros((2*ng, 2*ng), complex)
    S[0:ng, 0:ng] = array[0, 0, :, :]
    S[ng:2*ng, 0:ng] = array[0, 1, :, :]
    S[0:ng, ng:2*ng] = array[1, 0, :, :]
    S[ng:2*ng, ng:2*ng] = array[1, 1, :, :]
    xi = np.linalg.svd(S, compute_uv=False)
    plw = treams.plane_wave(kpar, 0, k0=k0[j], basis=pwb, material=1, modetype="down", poltype="parity")
    return j, i, SMat.tr(plw)[1], 1/xi[0]

results = Parallel(n_jobs=-1)(delayed(compute_svd)(i, j) for i in range(len(theta)) for j in range(len(wavelengths)))
refl = np.zeros((len(wavelengths), len(theta)))
xi1 = np.zeros((len(wavelengths), len(theta)))

for j, i, result1, result2 in results:
    refl[j, i] = result1
    xi1[j, i] = result2

Theta, Wavelengths = np.meshgrid(theta, wavelengths)
mpl.rcParams['pdf.fonttype'] = 42 
fs = 12

fig, ax = plt.subplots(2, 1)
mesh = ax[0].pcolormesh(Theta, Wavelengths, refl, cmap='YlOrBr', vmin=0, vmax=1, rasterized=True)
ax[0].plot(theta, (np.abs(np.sin(theta/180*np.pi)) + nsub) * L, linestyle="--", color="white") 
cbar = fig.colorbar(mesh, ax=ax[0])
ax[0].set_ylabel("$\\lambda$ (nm)", fontsize=fs)
cbar.set_label('Reflectance', fontsize=fs)
ax[0].tick_params(axis='x', bottom=False, labelbottom=False)
ax[0].tick_params(labelsize=fs)
cbar.ax.tick_params(labelsize=fs)

mesh = ax[1].pcolormesh(Theta, Wavelengths, xi1, cmap='Greens', norm=LogNorm(), rasterized=True)
ax[1].plot(theta, (np.abs(np.sin(theta/180*np.pi)) + nsub) * L, linestyle="--", color="white")  
cbar = fig.colorbar(mesh, ax=ax[1])
ax[1].set_ylabel("$\\lambda$ (nm)", fontsize=fs)
cbar.set_label('$\\xi_1^{-1}$', fontsize=fs)
ax[1].set_xlabel("Angle of incidence (deg)", fontsize=fs)
ax[1].tick_params(labelsize=fs)
cbar.ax.tick_params(labelsize=fs)
plt.show()
