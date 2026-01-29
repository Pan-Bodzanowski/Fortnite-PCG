import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, uniform_filter 
import os


def calculate_slope(data, moisture_data, slope_weight, river_threshold, river_bank_size):
    gy, gx = np.gradient(data)
    slope_map = np.sqrt(gx**2 + gy**2)
    slope_map = slope_map / (np.max(slope_map) + 1e-6)
    
    # filter area near rivers to avoid weird shapes
    is_river = moisture_data > river_threshold
    river_zone = binary_dilation(is_river, iterations=river_bank_size)
    avg_slope = uniform_filter(slope_map, size=river_bank_size * 4)
    slope_map[river_zone] = avg_slope[river_zone] * 0.5 
    
    return slope_map * slope_weight

def generate_biome_map(data, moisture_data, num_biomes=10, sea_level=0.5, 
                       height_weight=1500.0, moisture_weight=800.0, 
                       slope_weight=4000.0, river_threshold=0.85, river_bank_size=5): # 3. Nowe argumenty
    
    height, width = data.shape
    y_coords, x_coords = np.indices((height, width))
    land_mask = data > sea_level
    
    slope_weighted = calculate_slope(data, moisture_data, slope_weight, river_threshold, river_bank_size)
    
    moisture = moisture_data

    features = np.column_stack((
        x_coords[land_mask], 
        y_coords[land_mask], 
        data[land_mask] * height_weight,
        moisture[land_mask] * moisture_weight,
        slope_weighted[land_mask]
    ))

    kmeans = KMeans(n_clusters=num_biomes, random_state=42, n_init=10)
    sample_size = min(20000, features.shape[0])
    sample_idx = np.random.choice(features.shape[0], sample_size, replace=False)
    
    kmeans.fit(features[sample_idx])
    land_labels = kmeans.predict(features)

    centers = []
    for i in range(num_biomes):
        cluster_heights = data[land_mask][land_labels == i]
        centers.append(np.mean(cluster_heights) if len(cluster_heights) > 0 else -1)

    sorted_indices = np.argsort(centers)
    rank_map = {old_label: new_rank + 1 for new_rank, old_label in enumerate(sorted_indices)}
    
    final_labels = np.array([rank_map[label] for label in land_labels])
    
    biome_map = np.zeros_like(data, dtype=int)
    biome_map[land_mask] = final_labels
    return biome_map

def visualize_biomes(biome_map, save_path='step3_biome_distribution_map.png'):
    plt.figure(figsize=(12, 9))
    num_biomes = int(np.max(biome_map))
    cmap = plt.get_cmap('tab20', num_biomes + 1)
    img = plt.imshow(biome_map, cmap=cmap, interpolation='nearest', vmin=-0.5, vmax=num_biomes + 0.5)
    plt.title(f'Biome Map', fontsize=15, pad=20)
    cbar = plt.colorbar(img, ticks=range(num_biomes + 1))
    cbar.set_label('Biome classification ID', rotation=270, labelpad=20)
    tick_labels = [f'ID {i} (Water)' if i == 0 else f'Biome {i}' for i in range(num_biomes + 1)]
    cbar.ax.set_yticklabels(tick_labels)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    terrain_file = 'step1_island_noise_map.npy'
    moisture_file = 'step2_moisture_map.npy'
    
    output_file = 'step3_biome_map.npy'
    image_output_file = 'step3_biome_distribution_map.png'
    
    NUM_BIOMES = 12
    SEA_LEVEL = 0.52
    HEIGHT_WEIGHT = 1500.0
    MOISTURE_WEIGHT = 750.0
    
    SLOPE_WEIGHT = 3000.0
    RIVER_THRESHOLD = 0.92
    RIVER_BANK_SIZE = 8

    if not os.path.exists(terrain_file) or not os.path.exists(moisture_file):
        print(f"Error: Missing input files ({terrain_file} or {moisture_file}). Run steps 1 and 2 first.")
        exit()

    data = np.load(terrain_file)
    moisture_data = np.load(moisture_file)
    
    biome_map = generate_biome_map(
        data, 
        moisture_data,
        num_biomes=NUM_BIOMES, 
        sea_level=SEA_LEVEL, 
        height_weight=HEIGHT_WEIGHT,
        moisture_weight=MOISTURE_WEIGHT,
        slope_weight=SLOPE_WEIGHT,
        river_threshold=RIVER_THRESHOLD,
        river_bank_size=RIVER_BANK_SIZE
    )

    np.save(output_file, biome_map)
    print(f"Biome map saved as {output_file}.")
    
    visualize_biomes(biome_map, save_path=image_output_file)