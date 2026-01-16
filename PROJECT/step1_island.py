import numpy as np
import matplotlib.pyplot as plt

def generate_perlin_noise(width, height, scale=10):
    def lerp(a, b, t):
        return a + t * (b - a)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(h, x, y):
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_x = grid_x / scale
    grid_y = grid_y / scale

    x0 = np.floor(grid_x).astype(int)
    y0 = np.floor(grid_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    sx = fade(grid_x - x0)
    sy = fade(grid_y - y0)

    perm = np.random.randint(0, 256, size=(256,))
    perm = np.stack([perm, perm]).flatten()

    h00 = perm[(x0 + perm[y0 % 256]) % 256]
    h01 = perm[(x0 + perm[y1 % 256]) % 256]
    h10 = perm[(x1 + perm[y0 % 256]) % 256]
    h11 = perm[(x1 + perm[y1 % 256]) % 256]

    n00 = gradient(h00, grid_x - x0, grid_y - y0)
    n01 = gradient(h01, grid_x - x0, grid_y - y1)
    n10 = gradient(h10, grid_x - x1, grid_y - y0)
    n11 = gradient(h11, grid_x - x1, grid_y - y1)

    nx0 = lerp(n00, n10, sx)
    nx1 = lerp(n01, n11, sx)
    nxy = lerp(nx0, nx1, sy)

    return nxy

def generate_fractal_noise(width, height, scale, octaves=3, persistence=0.5, lacunarity=2.0):
    final_noise = np.zeros((height, width))
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for i in range(octaves):
        layer = generate_perlin_noise(width, height, scale * frequency)
        final_noise += layer * amplitude
        
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return final_noise / max_value

def add_random_mountains(terrain, width, height, num_clusters=2, peaks_per_cluster=10, max_extra_height=0.2):
    mountain_layer = np.zeros((height, width))
    
    land_indices = np.argwhere(terrain > np.percentile(terrain, 65))
    if len(land_indices) == 0: return mountain_layer

    for _ in range(num_clusters):
        cy, cx = land_indices[np.random.choice(len(land_indices))]
        
        for _ in range(peaks_per_cluster):
            offset_y = np.random.randint(-120, 120)
            offset_x = np.random.randint(-120, 120)
            px, py = np.clip(cx + offset_x, 0, width-1), np.clip(cy + offset_y, 0, height-1)
            
            y_grid, x_grid = np.ogrid[:height, :width]
            radius = np.random.randint(150, 350)
            dist_sq = (x_grid - px)**2 + (y_grid - py)**2
            mask = np.exp(-dist_sq / (2 * (radius/3)**2))
            
            scale_mnt = np.random.randint(15, 30)
            mnt_noise = generate_fractal_noise(width, height, scale=scale_mnt)
            mnt_noise = (mnt_noise - mnt_noise.min()) / (mnt_noise.max() - mnt_noise.min())
            
            mnt_noise = (1.0 - np.abs(mnt_noise - 0.5) * 2)**2
            
            mountain_layer = np.maximum(mountain_layer, mask * mnt_noise)

    return mountain_layer * max_extra_height

import matplotlib.colors as mcolors

def visualize_island(binary_noise, noise_terrain, sea_level):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax[0].imshow(binary_noise, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax[0].set_title(f'Binary Map ({sea_level=})')
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)

    colors = [
        (0.0, "darkblue"),
        (sea_level, "blue"),
        (sea_level, "khaki"),
        (sea_level + 0.1, "forestgreen"),
        (0.8, "saddlebrown"),
        (1.0, "white")]
    
    custom_terrain = mcolors.LinearSegmentedColormap.from_list("island_map", colors)

    im2 = ax[1].imshow(noise_terrain, cmap=custom_terrain, interpolation='bilinear', vmin=0, vmax=1)
    ax[1].set_title(f'Terrain Map ({sea_level=})')
    
    cbar = fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.ax.axhline(sea_level, color='red', linewidth=2)

    plt.tight_layout()
    plt.show()

W = H = 1024
S = 70
SEA_LEVEL = 0.4

y = np.arange(H)[:, None] 
x = np.arange(W)[None, :] 

p = 2.5
dist = abs(x - W//2)**p + abs(y - H//2)**p

mn, mx = -0.5, 5
gradient = dist / dist.max() * (mx - mn) + mn

raw_noise = generate_fractal_noise(W, H, S)

noise_terrain = raw_noise - (gradient * 0.5)
print(noise_terrain.shape)
noise_terrain = (noise_terrain - noise_terrain.min()) / (noise_terrain.max() - noise_terrain.min()) * 0.6
noise_terrain += add_random_mountains(noise_terrain, W, H)

binary_noise = (noise_terrain > SEA_LEVEL).astype(int)

np.save('island_noise_map.npy', noise_terrain)
visualize_island(binary_noise, noise_terrain, SEA_LEVEL)
