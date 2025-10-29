import json
import math

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
temp_dists = np.array(json.load(open("results/temperature_Au.json")))
nx = round(math.sqrt(temp_dists.shape[1]))
ny = nx
temp_dists = temp_dists.reshape(temp_dists.shape[0], nx, ny)
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
levels = np.linspace(temp_dists.min(), temp_dists.max(), 50)
file_paths = []
for i, temp_dist in tqdm(enumerate(temp_dists), total=temp_dists.shape[0]):
    _, ax = plt.subplots(figsize=(5, 3))
    ax.contourf(X, Y, temp_dist, levels=levels)
    ax.set_title(f"{1e-2 * i:.2f} [s]")
    ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    file_path = f"results/dist{i:0>3}.png"
    plt.savefig(file_path, bbox_inches="tight")
    file_paths.append(file_path)
    plt.close()

imageio.mimsave("results/dist.gif", [imageio.imread(fp) for fp in file_paths], duration=0.05, loop=0)
