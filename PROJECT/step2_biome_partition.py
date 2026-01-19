import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load the heightmap data
data = np.load('island_noise_map.npy')
height, width = data.shape

# 2. Parameters
num_biomes = 10 
sea_level = 0.5
# Higher height_weight = biomes follow elevation (rings)
# Lower height_weight = biomes follow location (patches)
height_weight = 1500.0 

y_coords, x_coords = np.indices((height, width))

land_mask = data > sea_level

features = np.column_stack((
    x_coords[land_mask], 
    y_coords[land_mask], 
    data[land_mask] * height_weight
))

print(f"Clustering {features.shape[0]} pixels...")

kmeans = KMeans(n_clusters=num_biomes, random_state=42, n_init=10)
sample_idx = np.random.choice(features.shape[0], 20000, replace=False)
kmeans.fit(features[sample_idx])
land_labels = kmeans.predict(features)

biome_map = np.zeros_like(data, dtype=int)

centers = []
for i in range(num_biomes):
    centers.append(np.mean(data[land_mask][land_labels == i]))

sorted_indices = np.argsort(centers)
rank_map = {old_label: new_rank + 1 for new_rank, old_label in enumerate(sorted_indices)}

final_labels = np.array([rank_map[label] for label in land_labels])
biome_map[land_mask] = final_labels

np.save('biome_map.npy', biome_map)

plt.figure(figsize=(10, 8))
plt.imshow(biome_map, cmap='terrain', interpolation='nearest')
plt.title(f'Spatially Consistent Biomes (Height Weight: {height_weight})')
plt.colorbar(label='Biome ID (0=Water)')
plt.axis('off')
plt.show()