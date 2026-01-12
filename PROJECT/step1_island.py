import numpy as np
from numpy.random import rand as r
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

W = H = 1024
S = 70

y = np.arange(H)[:, None] 
x = np.arange(W)[None, :] 

p = 2.5
dist = abs(x - W//2)**p + abs(y - H//2)**p

# dist = np.maximum(abs(x - W//2), abs(y - H//2))

mn, mx = -0.5, 3
gradient = dist / dist.max() * (mx - mn) + mn

noise = generate_perlin_noise(W, H, S)
noise = (noise > gradient).astype(int)

# Display using matplotlib without stretching pixels
plt.figure(figsize=(5, 4))
plt.imshow(noise, cmap='gray', interpolation='nearest', aspect='equal', vmin=0, vmax=1)
plt.title(f'Perlin Binary ({W}x{H}, scale={S})')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Binary value (0 or 1)')
plt.tight_layout()
plt.show()
