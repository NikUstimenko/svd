import numpy as np
import treams
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

c_const = 299792458
radius = 0.5e-2
L = 3e-2
materials = [treams.Material(2.1, 1, 0), treams.Material(1, 1, 0)]
freq = np.linspace(9.4e9, 9.6e9, 401) + 0.00024e9
k0 = 2 * np.pi * freq / c_const
distance = np.linspace(7e-2, 10e-2, 401)
kpar = np.array([0, 0])
kz = 0
mmax = 2
pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, [[L, 0], [0, L]], 1.01 * 2 * np.pi / L)
      
def compute_svd(i, j):
    rod = treams.TMatrixC.cylinder(kz, mmax, k0[j], radius, materials)
    rod = rod.changepoltype()
    positions = np.array([[0, -0.5 * distance[i], 0], [0, 0.5 * distance[i], 0]])
    Teff = treams.TMatrixC.cluster([rod, rod], positions).latticeinteraction.solve(L, kpar[0])
    sigma = np.linalg.svd(np.array(Teff), compute_uv=False)
    SMat = treams.SMatrices.from_array(Teff, pwb)
    array = np.array(SMat)
    ng = int(np.sqrt(np.size(array[0, 0, :, :])))
    S = np.zeros((2*ng, 2*ng), complex)
    S[0:ng, 0:ng] = array[0, 0, :, :]
    S[ng:2*ng, 0:ng] = array[0, 1, :, :]
    S[0:ng, ng:2*ng] = array[1, 0, :, :]
    S[ng:2*ng, ng:2*ng] = array[1, 1, :, :]
    xi = np.linalg.svd(S, compute_uv=False)
    plw = treams.plane_wave(kpar, 1, k0=k0[j], basis=pwb, 
                            material=treams.Material(1, 1, 0), modetype="down", poltype="parity")
    return j, i, SMat.tr(plw)[1], 1/sigma[0], 1/xi[0]

results = Parallel(n_jobs=-1)(delayed(compute_svd)(i, j) for i in range(len(distance)) for j in range(len(freq)))
refl = np.zeros((len(distance), len(freq)))
sigma1 = np.zeros((len(distance), len(freq)))
xi1 = np.zeros((len(distance), len(freq)))

for j, i, result1, result2, result3 in results:
    refl[j, i] = result1
    sigma1[j, i] = result2
    xi1[j, i] = result3

D, Freq = np.meshgrid(distance, freq)
mpl.rcParams['pdf.fonttype'] = 42 
fs = 12

fig, ax = plt.subplots(3, 1)
mesh = ax[0].pcolormesh(D * 1e2, c_const / Freq * 1e2, refl, cmap='YlOrBr', vmin=0, vmax=1, rasterized=True)
ax[0].plot(distance * 1e2, c_const / 9.48324e9 * 1e2 * np.ones(len(distance)), linestyle="--", color="blue")  
cbar = fig.colorbar(mesh, ax=ax[0])
ax[0].set_ylabel("Wavelength (cm)", fontsize=fs)
cbar.set_label('Reflectance', fontsize=fs)
ax[0].tick_params(axis='x', bottom=False, labelbottom=False)
ax[0].tick_params(labelsize=fs)
cbar.ax.tick_params(labelsize=fs)

mesh = ax[1].pcolormesh(D * 1e2, c_const / Freq * 1e2, sigma1, cmap='Blues', norm=LogNorm(), rasterized=True)
ax[1].plot(distance * 1e2, c_const / 9.48324e9 * 1e2 * np.ones(len(distance)), linestyle="--", color="blue") 
cbar = fig.colorbar(mesh, ax=ax[1])
ax[1].set_ylabel("Wavelength (cm)", fontsize=fs)
cbar.set_label('$\\sigma_1^{-1}$', fontsize=fs)
ax[1].tick_params(axis='x', bottom=False, labelbottom=False)
ax[1].tick_params(labelsize=fs)
cbar.ax.tick_params(labelsize=fs)

mesh = ax[2].pcolormesh(D * 1e2, c_const / Freq * 1e2, xi1, cmap='Greens', norm=LogNorm(), rasterized=True)
ax[2].plot(distance * 1e2, c_const / 9.48324e9 * 1e2 * np.ones(len(distance)), linestyle="--", color="blue") 
cbar = fig.colorbar(mesh, ax=ax[2])
ax[2].set_ylabel("Wavelength (cm)", fontsize=fs)
cbar.set_label('$\\xi_1^{-1}$', fontsize=fs)
ax[2].set_xlabel("Distance (cm)", fontsize=fs)
ax[2].tick_params(labelsize=fs)
cbar.ax.tick_params(labelsize=fs)
plt.show()
