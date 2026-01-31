import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import binary_dilation, generate_binary_structure

TERRAIN_FILE = 'step1_island_noise_map.npy'
BIOME_FILE = 'step3_biome_map.npy'

OUTPUT_TERRAIN = 'step4_terrain_with_cities.npy'
OUTPUT_CITIES_MASK = 'step4_cities_mask.npy'
OUTPUT_IMAGE = 'step4_final_map_clean.png'

SEA_LEVEL = 0.52
NUM_CITIES = 14
CITY_SIZE_RATIO = 25


def get_land_biome_indices(biome_map, terrain, sea_level):
    land_mask = terrain > sea_level
    masked_biomes = biome_map.copy()
    masked_biomes[~land_mask] = -1

    unique_ids = np.unique(masked_biomes)
    unique_ids = unique_ids[unique_ids != -1]

    biome_indices = {}
    for bid in unique_ids:
        y, x = np.where(masked_biomes == bid)
        if len(y) > 0:
            biome_indices[bid] = np.column_stack((y, x))
    return biome_indices, unique_ids


def check_conditions(terrain, r, c, half_size, sea_level, existing_cities, min_dist_sq,
                     max_variance=0.02, max_height=0.75, biome_map=None, target_biome_id=None):
    H, W = terrain.shape
    r_min, r_max = r - half_size, r + half_size
    c_min, c_max = c - half_size, c + half_size

    if r_min < 0 or r_max > H or c_min < 0 or c_max > W:
        return False, None

    patch = terrain[r_min:r_max, c_min:c_max]

    if np.min(patch) <= sea_level:
        return False, None
    if np.std(patch) > max_variance:
        return False, None
    if np.max(patch) > max_height:
        return False, None

    for er, ec in existing_cities:
        if (r - er)**2 + (c - ec)**2 < min_dist_sq:
            return False, None

    if biome_map is not None and target_biome_id is not None:
        biome_patch = biome_map[r_min:r_max, c_min:c_max]
        unique_biomes_in_city = np.unique(biome_patch)
        if len(unique_biomes_in_city) > 1:
            return False, None
        if unique_biomes_in_city[0] != target_biome_id:
            return False, None

    return True, (r_min, r_max, c_min, c_max)


def find_city_locations(terrain, biome_map, sea_level, num_cities, attempts_per_biome=40):
    H, W = terrain.shape
    city_size = min(H, W) // CITY_SIZE_RATIO
    half_size = city_size // 2
    min_dist_sq = (city_size * 3.5) ** 2

    cities_centers = []
    biome_indices_dict, biome_ids = get_land_biome_indices(
        biome_map, terrain, sea_level)

    pools = {bid: np.random.permutation(coords)
             for bid, coords in biome_indices_dict.items()}
    pool_pointers = {bid: 0 for bid in biome_ids}

    cities_per_biome = {bid: 0 for bid in biome_ids}
    biome_has_space = {bid: True for bid in biome_ids}
    rr_index = 0

    # Round robin
    for _ in range(num_cities):
        placed = False
        for _ in range(len(biome_ids)):
            bid = biome_ids[rr_index]
            rr_index = (rr_index + 1) % len(biome_ids)

            if not biome_has_space[bid] or cities_per_biome[bid] > (len(cities_centers) / len(biome_ids) + 1):
                continue

            start_ptr = pool_pointers[bid]
            end_ptr = min(start_ptr + attempts_per_biome, len(pools[bid]))

            for i in range(start_ptr, end_ptr):
                r, c = pools[bid][i]
                valid, _ = check_conditions(terrain, r, c, half_size, sea_level, cities_centers,
                                            min_dist_sq, biome_map=biome_map, target_biome_id=bid)
                if valid:
                    cities_centers.append((r, c))
                    cities_per_biome[bid] += 1
                    pool_pointers[bid] = i + 1
                    placed = True
                    break

            if placed:
                break
            pool_pointers[bid] = end_ptr
            if pool_pointers[bid] >= len(pools[bid]):
                biome_has_space[bid] = False

    # Random Fallback
    while len(cities_centers) < num_cities:
        r, c = np.random.randint(
            half_size, H-half_size), np.random.randint(half_size, W-half_size)
        valid, _ = check_conditions(terrain, r, c, half_size, sea_level, cities_centers,
                                    min_dist_sq)
        if valid:
            cities_centers.append((r, c))

    return cities_centers, half_size


def apply_city_features(terrain, cities_centers, half_size, flattening_factor=0.8):
    modified_terrain = terrain.copy()
    cities_mask = np.zeros_like(terrain, dtype=int)

    for idx, (r, c) in enumerate(cities_centers):
        r0, r1, c0, c1 = r - half_size, r + half_size, c - half_size, c + half_size

        cities_mask[r0:r1, c0:c1] = idx + 1

        patch = modified_terrain[r0:r1, c0:c1]
        target_h = np.mean(patch)
        modified_terrain[r0:r1, c0:c1] = (
            patch * (1 - flattening_factor)) + (target_h * flattening_factor)

    return modified_terrain, cities_mask


def get_boundaries(data_map):
    grad_x = np.diff(data_map, axis=1, append=data_map[:, -1:]) != 0
    grad_y = np.diff(data_map, axis=0, append=data_map[-1:, :]) != 0
    edges = grad_x | grad_y
    struct = generate_binary_structure(2, 1)
    return binary_dilation(edges, structure=struct, iterations=1)


def visualize_cities(terrain, biome_map, cities_mask, sea_level, save_path='step4_final_map_clean.png'):
    H, W = terrain.shape
    fig, ax = plt.subplots(figsize=(12, 12))

    colors = [(0.0, "darkblue"), (sea_level, "blue"), (sea_level, "khaki"),
              (sea_level + 0.1, "forestgreen"), (0.8, "saddlebrown"), (1.0, "white")]
    cmap_terrain = mcolors.LinearSegmentedColormap.from_list("island", colors)
    ax.imshow(terrain, cmap=cmap_terrain,
              interpolation='bilinear', vmin=0, vmax=1)

    land_biomes = np.where(terrain > sea_level, biome_map, -1)
    biome_edges = get_boundaries(land_biomes)
    biome_edge_mask = biome_edges & (terrain > sea_level)

    overlay_biomes = np.zeros((H, W, 4))
    overlay_biomes[biome_edge_mask] = [0.8, 0.0, 1.0, 0.6]
    ax.imshow(overlay_biomes)

    city_binary = cities_mask > 0
    city_edges = get_boundaries(city_binary.astype(int))
    thick_city_edges = binary_dilation(city_edges, iterations=1)

    overlay_cities = np.zeros((H, W, 4))
    overlay_cities[city_binary] = [0.0, 0.0, 0.0, 0.4]
    overlay_cities[thick_city_edges] = [0.8, 1.0, 0.0, 1.0]

    ax.imshow(overlay_cities)
    ax.set_title(f'Cities map with biome borders', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    raw_terrain = np.load(TERRAIN_FILE)
    biome_map = np.load(BIOME_FILE)

    centers, h_size = find_city_locations(
        raw_terrain, biome_map, SEA_LEVEL, NUM_CITIES)
    final_terrain, cities_mask = apply_city_features(
        raw_terrain, centers, h_size)

    # 3. Save & Visualize
    np.save(OUTPUT_TERRAIN, final_terrain)
    print(f"Modified terrain saved to {OUTPUT_TERRAIN}")

    np.save(OUTPUT_CITIES_MASK, cities_mask)
    print(f"City IDs mask saved to {OUTPUT_CITIES_MASK}")

    visualize_cities(final_terrain, biome_map, cities_mask,
                     SEA_LEVEL, save_path=OUTPUT_IMAGE)
    print(f"Map visualization saved to {OUTPUT_IMAGE}")
