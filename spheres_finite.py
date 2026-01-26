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
L = 1037.77895
lattice = treams.Lattice.square(L)
wavelengths = np.linspace(1450, 1470, 401) + 0.01
k0 = 2 * np.pi / wavelengths
N = np.linspace(3, 23, 11)

def generate_lattice_coordinates(n, spacing):
    n = int(n)
    if n == 1:
        return np.array([[0, 0, 0]])
    points_x = np.linspace(0, (n - 1) * spacing[0], n)
    points_y = np.linspace(0, (n - 1) * spacing[1], n)
    x, y = np.meshgrid(points_x, points_y)
    z = np.full_like(x, 0)
    coordinates = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    coordinates -= np.mean(coordinates, axis=0)
    return coordinates

def compute_svd(i, j):
    positions = generate_lattice_coordinates(N[i], [L, L])
    k0 = 2 * np.pi / wavelengths[j]
    sphere = [treams.TMatrix.sphere(lmax, k0, radius, materials, poltype="parity")] * int(N[i] * N[i])
    Teff = treams.TMatrix.cluster(sphere, positions).interaction.solve()
    Teff = np.array(Teff)
    sigma = np.linalg.svd(Teff, compute_uv=False)
    return i, j, 1/sigma[0]

results = Parallel(n_jobs=-1)(delayed(compute_svd)(i, j) for i in range(len(N)) for j in range(len(wavelengths)))
sigma1 = np.zeros((len(N), len(wavelengths)))
for i, j, result in results:
    sigma1[i, j] = result

mpl.rcParams['pdf.fonttype'] = 42 
fs = 18

Wavelength_R = 1459.16

fig, ax = plt.subplots()
cax = ax.pcolormesh(wavelengths - Wavelength_R, N, sigma1, norm=LogNorm(), cmap='Blues', shading='auto', rasterized=True)
ax.set_ylabel('Number of particles per side $N$', fontsize=fs)
ax.set_xlabel("$\\Delta \\lambda$ (nm)", fontsize=fs)
ax.set_ylim([int(np.min(N)) - 1, int(np.max(N)) + 1])
ax.set_yticks(N.astype(int))
ax.set_yticklabels(N.astype(int))
ax.tick_params(labelsize=fs)
y_ticks = ax.get_yticks()
for y_tick in y_ticks:
    if y_tick > np.min(y_ticks) and y_tick < np.max(y_ticks):
        ax.axhline(y=y_tick + 1, color='gray', linewidth=0.5)
        ax.axhline(y=y_tick - 1, color='gray', linewidth=0.5)
ax.plot(np.zeros(len(N)), N, linestyle="--", color='red') 
cbar = fig.colorbar(cax)
cbar.set_label('$\\sigma_1^{-1}$', fontsize=fs)
cbar.ax.tick_params(labelsize=fs)
plt.show()

row_mins = np.min(sigma1, axis=1)
fig, ax = plt.subplots()
ax.scatter(N, row_mins)
ax.set_xlabel('Number of particles per side $N$', fontsize=fs)
ax.set_ylabel('${\\rm min}_{\\lambda} \\sigma_1^{-1}$', fontsize=fs)
ax.tick_params(labelsize=fs)
ax.set_yscale("log")
plt.show()