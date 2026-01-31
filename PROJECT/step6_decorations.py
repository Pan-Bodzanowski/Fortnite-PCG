# step6_biomes_and_decor.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation
from pathlib import Path

from step1_island import fractal_noise  # still used in earlier steps

# folders
DATA_DIR = Path('data')
IMAGES_DIR = Path('images')
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# input files
TERRAIN_RAW_FILE = DATA_DIR / 'step1_island_noise_map.npy'
MOISTURE_FILE = DATA_DIR / 'step2_moisture_map.npy'
BIOME_PARTITION_FILE = DATA_DIR / 'step3_biome_map.npy'
TERRAIN_WITH_CITIES_FILE = DATA_DIR / 'step4_terrain_with_cities.npy'

# outputs
OUTPUT_BIOME_TYPE_MAP = DATA_DIR / 'step6_biome_type_map.npy'
OUTPUT_TREE_MASK = DATA_DIR / 'step6_tree_mask.npy'
OUTPUT_IMAGE = IMAGES_DIR / 'step6_biomes_trees.png'

SEA_LEVEL = 0.52

# ---------------------------------------------------------------------
# CONFIG: biome semantics
# ---------------------------------------------------------------------

BIOME_NAMES = [
    "Grasslands",
    "Desert",
    "Snow",
    "Beach",
    "Forest",
    "Swamp",
    "Boreal Tundra",
]

DETERMINISTIC_BIOME_ASSIGNMENT = True

# ---------------------------------------------------------------------
# CONFIG: tree densities and sizes per biome
# ---------------------------------------------------------------------
TREE_CONFIG = {
    "Forest":        {"density": 0.020, "radius": 5},
    "Grasslands":    {"density": 0.002, "radius": 4},
    "Swamp":         {"density": 0.010, "radius": 4},
    "Boreal Tundra": {"density": 0.004, "radius": 4},
    "Snow":          {"density": 0.002, "radius": 3},
    "Desert":        {"density": 0.000, "radius": 3},
    "Beach":         {"density": 0.000, "radius": 3},
}

# ---------------------------------------------------------------------
# Helper functions (unchanged except valley removed)
# ---------------------------------------------------------------------

def compute_biome_stats(terrain, moisture, biome_map, sea_level):
    H, W = terrain.shape
    land_mask = terrain > sea_level
    water_mask = ~land_mask

    from scipy.ndimage import distance_transform_edt
    dist_to_water = distance_transform_edt(~water_mask)

    edge_mask = np.zeros_like(terrain, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    dist_to_edge = distance_transform_edt(~edge_mask)

    biome_ids = np.unique(biome_map)
    stats = {}

    for bid in biome_ids:
        if bid == 0:
            continue
        mask = (biome_map == bid) & land_mask
        if not np.any(mask):
            continue

        stats[bid] = {
            "mean_height": float(np.mean(terrain[mask])),
            "mean_moisture": float(np.mean(moisture[mask])),
            "mean_dist_water": float(np.mean(dist_to_water[mask])),
            "mean_dist_edge": float(np.mean(dist_to_edge[mask])),
            "area": int(mask.sum()),
        }

    return stats


def compute_biome_adjacency(biome_map):
    H, W = biome_map.shape
    adj = {bid: set() for bid in np.unique(biome_map)}

    for r in range(H - 1):
        for c in range(W - 1):
            a = biome_map[r, c]
            b = biome_map[r + 1, c]
            c2 = biome_map[r, c + 1]
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
            if a != c2:
                adj[a].add(c2)
                adj[c2].add(a)

    return adj


def assign_biomes_deterministic(terrain, moisture, biome_map, sea_level):
    stats = compute_biome_stats(terrain, moisture, biome_map, sea_level)
    adj = compute_biome_adjacency(biome_map)

    biome_ids = sorted(stats.keys())
    available_biomes = BIOME_NAMES.copy()
    biome_id_to_name = {}
    assigned_ids = set()

    def pick_best(candidates, score_fn, exclude):
        best_id = None
        best_score = None
        for bid in candidates:
            if bid in exclude:
                continue
            s = score_fn(stats[bid])
            if best_score is None or s > best_score:
                best_score = s
                best_id = bid
        return best_id

    # Beach
    if "Beach" in available_biomes:
        def beach_score(s):
            return -s["mean_height"] - 0.5 * s["mean_dist_water"]
        bid = pick_best(biome_ids, beach_score, assigned_ids)
        if bid is not None:
            biome_id_to_name[bid] = "Beach"
            assigned_ids.add(bid)
            available_biomes.remove("Beach")

    # Snow
    if "Snow" in available_biomes:
        def snow_score(s):
            return 2*s["mean_height"] + s["mean_dist_water"] + 0.5*s["mean_dist_edge"]
        bid = pick_best(biome_ids, snow_score, assigned_ids)
        if bid is not None:
            biome_id_to_name[bid] = "Snow"
            assigned_ids.add(bid)
            available_biomes.remove("Snow")
            snow_id = bid
        else:
            snow_id = None
    else:
        snow_id = None

    # Desert
    if "Desert" in available_biomes:
        def desert_score(s):
            return -s["mean_moisture"] + 0.2*s["mean_height"]
        candidates = [
            bid for bid in biome_ids
            if bid not in assigned_ids and (snow_id is None or snow_id not in adj[bid])
        ]
        bid = pick_best(candidates, desert_score, assigned_ids)
        if bid is not None:
            biome_id_to_name[bid] = "Desert"
            assigned_ids.add(bid)
            available_biomes.remove("Desert")

    # Swamp
    if "Swamp" in available_biomes:
        def swamp_score(s):
            return 1.5*s["mean_moisture"] - 0.5*s["mean_height"] - 0.5*s["mean_dist_water"]
        bid = pick_best(biome_ids, swamp_score, assigned_ids)
        if bid is not None:
            biome_id_to_name[bid] = "Swamp"
            assigned_ids.add(bid)
            available_biomes.remove("Swamp")

    # Forest
    if "Forest" in available_biomes:
        def forest_score(s):
            return 1.5*s["mean_moisture"] - 0.3*abs(s["mean_height"] - 0.55)
        bid = pick_best(biome_ids, forest_score, assigned_ids)
        if bid is not None:
            biome_id_to_name[bid] = "Forest"
            assigned_ids.add(bid)
            available_biomes.remove("Forest")

    # Grasslands
    if "Grasslands" in available_biomes:
        def grass_score(s):
            return -abs(s["mean_moisture"] - 0.5) - 0.2*abs(s["mean_height"] - 0.5)
        bid = pick_best(biome_ids, grass_score, assigned_ids)
        if bid is not None:
            biome_id_to_name[bid] = "Grasslands"
            assigned_ids.add(bid)
            available_biomes.remove("Grasslands")

    # Boreal Tundra
    if "Boreal Tundra" in available_biomes:
        def tundra_score(s):
            return s["mean_height"] - 0.3*abs(s["mean_moisture"] - 0.4)
        bid = pick_best(biome_ids, tundra_score, assigned_ids)
        if bid is not None:
            biome_id_to_name[bid] = "Boreal Tundra"
            assigned_ids.add(bid)
            available_biomes.remove("Boreal Tundra")

    # Remaining
    remaining = [bid for bid in biome_ids if bid not in assigned_ids]
    if not available_biomes:
        available_biomes = ["Grasslands"]

    for i, bid in enumerate(remaining):
        biome_id_to_name[bid] = available_biomes[i % len(available_biomes)]

    return biome_id_to_name


def assign_biomes_random(biome_map):
    biome_ids = np.unique(biome_map)
    biome_ids = biome_ids[biome_ids != 0]
    rng = np.random.default_rng()
    names = BIOME_NAMES.copy()
    rng.shuffle(names)
    return {bid: names[i % len(names)] for i, bid in enumerate(biome_ids)}


def build_biome_type_map(biome_map, biome_id_to_name):
    biome_name_to_type_id = {name: i+1 for i, name in enumerate(BIOME_NAMES)}
    biome_type_map = np.zeros_like(biome_map, dtype=int)
    for bid, name in biome_id_to_name.items():
        biome_type_map[biome_map == bid] = biome_name_to_type_id[name]
    return biome_type_map, biome_name_to_type_id


def generate_tree_mask(biome_type_map, biome_name_to_type_id, terrain):
    H, W = terrain.shape
    tree_mask = np.zeros((H, W), dtype=int)
    rng = np.random.default_rng()

    for biome_name, cfg in TREE_CONFIG.items():
        density = cfg["density"]
        radius = cfg["radius"]
        if density <= 0:
            continue

        type_id = biome_name_to_type_id.get(biome_name)
        coords = np.column_stack(np.where(biome_type_map == type_id))
        if len(coords) == 0:
            continue

        n_trees = int(len(coords) * density)
        if n_trees <= 0:
            continue

        chosen = rng.choice(len(coords), size=n_trees, replace=False)
        centers = coords[chosen]

        yb, xb = np.ogrid[-radius:radius+1, -radius:radius+1]
        brush = (xb**2 + yb**2) <= radius**2

        for (r, c) in centers:
            r0 = max(0, r - radius)
            r1 = min(H, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(W, c + radius + 1)

            br_r0 = r0 - (r - radius)
            br_c0 = c0 - (c - radius)
            br_r1 = br_r0 + (r1 - r0)
            br_c1 = br_c0 + (c1 - c0)

            tree_mask[r0:r1, c0:c1][brush[br_r0:br_r1, br_c0:br_c1]] = 1

    return tree_mask


def build_biome_colormap():
    return {
        "Grasslands":    (0.55, 0.80, 0.35, 0.9),
        "Desert":        (0.93, 0.80, 0.45, 0.9),
        "Snow":          (1.00, 1.00, 1.00, 0.9),
        "Beach":         (0.96, 0.90, 0.65, 0.9),
        "Forest":        (0.10, 0.45, 0.15, 0.9),
        "Swamp":         (0.20, 0.35, 0.25, 0.9),
        "Boreal Tundra": (0.60, 0.75, 0.80, 0.9),
    }


def visualize_all(terrain, biome_type_map, biome_name_to_type_id,
                  tree_mask, sea_level, save_path):

    H, W = terrain.shape
    fig, ax = plt.subplots(figsize=(12, 12))

    colors = [
        (0.0, "darkblue"),
        (sea_level, "blue"),
        (sea_level, "khaki"),
        (sea_level + 0.1, "forestgreen"),
        (0.8, "saddlebrown"),
        (1.0, "white"),
    ]
    cmap_terrain = mcolors.LinearSegmentedColormap.from_list("island", colors)
    ax.imshow(terrain, cmap=cmap_terrain, interpolation='bilinear', vmin=0, vmax=1)

    biome_colors = build_biome_colormap()
    overlay = np.zeros((H, W, 4))
    for biome_name, type_id in biome_name_to_type_id.items():
        overlay[biome_type_map == type_id] = biome_colors[biome_name]
    ax.imshow(overlay)

    tree_overlay = np.zeros((H, W, 4))
    tree_overlay[tree_mask > 0] = [0.0, 0.25, 0.0, 0.9]
    ax.imshow(tree_overlay)

    ax.set_title("Biomes and Trees", fontsize=16)
    ax.axis('off')

    legend_patches = [
        Patch(color=color, label=name)
        for name, color in biome_colors.items()
    ]
    legend_patches.append(Patch(color=(0.0, 0.25, 0.0, 0.9), label="Trees"))

    ax.legend(handles=legend_patches, loc='lower left',
              bbox_to_anchor=(0.01, 0.01), fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Final biome+tree map saved to {save_path}")
    plt.show()


def main():
    terrain_raw = np.load(TERRAIN_RAW_FILE)
    moisture = np.load(MOISTURE_FILE)
    biome_map = np.load(BIOME_PARTITION_FILE)
    terrain_with_cities = np.load(TERRAIN_WITH_CITIES_FILE)

    base_terrain = terrain_with_cities

    if DETERMINISTIC_BIOME_ASSIGNMENT:
        biome_id_to_name = assign_biomes_deterministic(
            terrain_raw, moisture, biome_map, SEA_LEVEL
        )
    else:
        biome_id_to_name = assign_biomes_random(biome_map)

    biome_type_map, biome_name_to_type_id = build_biome_type_map(
        biome_map, biome_id_to_name
    )

    tree_mask = generate_tree_mask(biome_type_map, biome_name_to_type_id,
                                   base_terrain)

    np.save(OUTPUT_BIOME_TYPE_MAP, biome_type_map)
    np.save(OUTPUT_TREE_MASK, tree_mask)

    visualize_all(base_terrain, biome_type_map, biome_name_to_type_id,
                  tree_mask, SEA_LEVEL, OUTPUT_IMAGE)

if __name__ == "__main__":
    main()
