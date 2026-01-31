import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import matplotlib.colors as mcolors


def generate_perlin_noise(width, height, scale=10):
    def lerp(a, b, t):
        return a + t * (b - a)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(h, x, y):
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[..., 0] * x + g[..., 1] * y

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


def fractal_noise(w, h, S, octaves, persistence):
    noise = np.zeros((h, w))
    freq, amp = 1.0, 1.0
    for _ in range(octaves):
        noise += amp * generate_perlin_noise(w, h, S / freq)
        freq *= 2
        amp *= persistence
    return noise


def create_radial_gradient(W, H, p=3, mn=0, mx=7):
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    xc = x - W // 2
    yc = y - H // 2
    dist = abs(xc)**p + abs(yc)**p
    return dist / dist.max() * (mx - mn) + mn


def apply_river_canyons(noise_terrain, W, H, S_RIV=500, RIV_BOTTOM=0.5, OCT_RIV=4, PERS_RIV=0.5, W_RIV=0.02, CANYON=50):
    river_noise = fractal_noise(W, H, S_RIV, OCT_RIV, PERS_RIV)
    river_mask = 1 - (abs(river_noise) < W_RIV).astype(int)

    dist = distance_transform_edt(river_mask)
    dist = np.minimum(np.array([CANYON]), dist)
    dist /= CANYON

    noise_terrain -= RIV_BOTTOM
    noise_terrain *= dist ** 0.5
    noise_terrain += RIV_BOTTOM
    return noise_terrain


def generate_island_terrain(W, H, S, OCT, PERS):
    gradient_mask = create_radial_gradient(W, H)
    raw_noise = fractal_noise(W, H, S, OCT, PERS)

    noise_terrain = raw_noise - (gradient_mask * 0.5)
    noise_terrain = (noise_terrain - noise_terrain.min()) / \
        (noise_terrain.max() - noise_terrain.min()) * 0.8

    noise_terrain = apply_river_canyons(noise_terrain, W, H)
    return noise_terrain


def visualize_island(noise_terrain, sea_level):
    binary_noise = (noise_terrain > sea_level).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax[0].imshow(binary_noise, cmap='gray',
                       interpolation='nearest', vmin=0, vmax=1)
    ax[0].set_title(f'Binary Map ({sea_level=})')
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)

    colors = [
        (0.0, "darkblue"),
        (sea_level, "blue"),
        (sea_level, "khaki"),
        (sea_level + 0.1, "forestgreen"),
        (0.8, "saddlebrown"),
        (1.0, "white")]

    custom_terrain = mcolors.LinearSegmentedColormap.from_list(
        "island_map", colors)

    im2 = ax[1].imshow(noise_terrain, cmap=custom_terrain,
                       interpolation='bilinear', vmin=0, vmax=1)
    ax[1].set_title(f'Terrain Map ({sea_level=})')

    cbar = fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.ax.axhline(sea_level, color='red', linewidth=2)

    plt.tight_layout()
    plt.savefig('step1_island_noise_map.png')
    plt.show()


def main():
    W = H = 1024
    S = 120
    SEA_LEVEL = 0.52
    OCT = 3
    PERS = 0.45

    noise_terrain = generate_island_terrain(W, H, S, OCT, PERS)

    np.save('step1_island_noise_map.npy', noise_terrain)
    visualize_island(noise_terrain, SEA_LEVEL)


if __name__ == "__main__":
    main()
