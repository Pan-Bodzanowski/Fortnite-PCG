import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from step1_island import fractal_noise

def calculate_moisture(data, sea_level, sigma_ocean=100, sigma_rivers=15, noise_strength=0.35):
    height, width = data.shape
    land_mask = data > sea_level
    
    moisture_ocean = gaussian_filter((data <= sea_level).astype(float), sigma=sigma_ocean)
    
    river_mask = (data <= sea_level) & (gaussian_filter(land_mask.astype(float), sigma=2) > 0)
    moisture_rivers = gaussian_filter(river_mask.astype(float), sigma=sigma_rivers) * 3.0
    
    dist_moisture = moisture_ocean + moisture_rivers
    
    if dist_moisture[land_mask].max() > 0:
        dist_moisture = dist_moisture / dist_moisture[land_mask].max()

    if noise_strength > 0:
        f_noise = fractal_noise(width, height, S=150, octaves=4, persistence=0.5)
        f_noise = (f_noise - f_noise.min()) / (f_noise.max() - f_noise.min())
        moisture_map = (1 - noise_strength) * dist_moisture + noise_strength * f_noise
    else:
        moisture_map = dist_moisture

    moisture_map[~land_mask] = 1.0 
    
    return np.clip(moisture_map, 0, 1)


def visualize_moisture(moisture_map, terrain_data, sea_level, save_path='step2_moisture_map.png'):
    land_mask = terrain_data > sea_level
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    ax[0].imshow(land_mask, cmap='gray', interpolation='nearest')
    ax[0].set_title('Land Mask (Context)')
    ax[0].axis('off')

    moisture_colors = [
        (0.0, "#8B4513"),
        (0.25, "#DEB887"),
        (0.5, "#9ACD32"),
        (0.75, "#20B2AA"),
        (1.0, "#000080") 
    ]
    moisture_cmap = mcolors.LinearSegmentedColormap.from_list("moisture_map", moisture_colors)

    im2 = ax[1].imshow(moisture_map, cmap=moisture_cmap, interpolation='bilinear', vmin=0, vmax=1)
    ax[1].set_title('Moisture Distribution Map')
    
    cbar = fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label('Moisture Level')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    INPUT_FILE = 'step1_island_noise_map.npy'
    
    OUTPUT_FILE = 'step2_moisture_map.npy'
    IMAGE_FILE = 'step2_moisture_map.png'
    
    SEA_LEVEL = 0.52
    
    try:
        terrain_data = np.load(INPUT_FILE)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {INPUT_FILE}. Uruchom najpierw kod z kroku 1.")
        exit()

    moisture = calculate_moisture(terrain_data, SEA_LEVEL, sigma_ocean=100, sigma_rivers=40, noise_strength=0.4)

    np.save(OUTPUT_FILE, moisture)
    visualize_moisture(moisture, terrain_data, SEA_LEVEL, save_path=IMAGE_FILE)
    print(f"Moisture map saved as {OUTPUT_FILE} and image as {IMAGE_FILE}.")