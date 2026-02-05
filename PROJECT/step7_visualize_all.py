import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation
from pathlib import Path
from step6_decorations import build_biome_colormap

DATA_DIR = Path('data')
IMAGES_DIR = Path('images')
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TERRAIN_FILE = DATA_DIR / 'step4_terrain_with_cities.npy'
CITIES_FILE = DATA_DIR / 'step4_cities_mask.npy'
ROADS_FILE = DATA_DIR / 'step5_roads_mask.npy'
BIOME_FILE = DATA_DIR / 'step6_biome_type_map.npy'
TREES_FILE = DATA_DIR / 'step6_tree_mask.npy'
BRIDGES_FILE = DATA_DIR / 'step6_bridge_mask.npy'

OUTPUT_IMAGE = IMAGES_DIR / 'step7_final_world_map.png'

SEA_LEVEL = 0.52

named_colors = build_biome_colormap()

BIOME_COLORS = {
    0: (0.0, 0.0, 0.5, 1.0),
    1: named_colors["Grasslands"],
    2: named_colors["Desert"],
    3: named_colors["Snow"],
    4: named_colors["Beach"],
    5: named_colors["Forest"],
    6: named_colors["Swamp"],
    7: named_colors["Boreal Tundra"],
}

COLOR_ROAD = "#ba9e7a"
COLOR_BRIDGE = "#2c2c2c"
COLOR_CITY = "#202020"
COLOR_TREE = "#003300"

COLOR_WATER_DEEP = "#1e4d8c"
COLOR_WATER_SHALLOW = "#4db6e3"

# opacity
CITY_ALPHA = 0.75
TREE_ALPHA = 0.80

BRIDGE_VISUAL_THICKNESS = 4


def create_colormap_from_dict(color_dict, max_id):
    colors_list = []
    for i in range(max_id + 1):
        c = color_dict.get(i, (0, 0, 0, 0))
        colors_list.append(c)
    return mcolors.ListedColormap(colors_list)


def blend_layers(background_rgb, overlay_color_hex, alpha):
    overlay_rgba = mcolors.to_rgba(overlay_color_hex)
    overlay_rgb = np.array(overlay_rgba)
    blended = background_rgb * (1 - alpha) + overlay_rgb * alpha
    blended[:, 3] = 1.0
    return blended


def render_map():
    terrain = np.load(TERRAIN_FILE)
    cities = np.load(CITIES_FILE)
    roads = np.load(ROADS_FILE)
    biomes = np.load(BIOME_FILE)
    trees = np.load(TREES_FILE)
    if BRIDGES_FILE.exists():
        bridges = np.load(BRIDGES_FILE)
    else:
        bridges = np.zeros_like(terrain)

    ls = LightSource(azdeg=315, altdeg=45)

    max_biome_id = int(np.max(biomes))
    if max_biome_id < 1:
        max_biome_id = 1

    biome_cmap = create_colormap_from_dict(BIOME_COLORS, max_biome_id)
    biome_norm = mcolors.Normalize(vmin=0, vmax=max_biome_id)
    biome_rgba = biome_cmap(biome_norm(biomes))

    rgb_map = ls.shade_rgb(biome_rgba, terrain,
                           vert_exag=150, blend_mode='overlay')

    is_water = terrain <= SEA_LEVEL

    water_depth = (SEA_LEVEL - terrain)
    water_depth[~is_water] = 0
    if water_depth.max() > 0:
        water_depth /= water_depth.max()

    c_deep = mcolors.to_rgb(COLOR_WATER_DEEP)
    c_shallow = mcolors.to_rgb(COLOR_WATER_SHALLOW)

    water_r = c_shallow[0] * (1 - water_depth) + c_deep[0] * water_depth
    water_g = c_shallow[1] * (1 - water_depth) + c_deep[1] * water_depth
    water_b = c_shallow[2] * (1 - water_depth) + c_deep[2] * water_depth
    water_a = np.ones_like(water_r)

    water_rgba = np.dstack((water_r, water_g, water_b, water_a))
    rgb_map[is_water] = water_rgba[is_water]

    tree_indices = trees > 0
    if np.any(tree_indices):
        rgb_map[tree_indices] = blend_layers(
            rgb_map[tree_indices], COLOR_TREE, TREE_ALPHA
        )

    if np.any(bridges > 0):
        thick_bridges_mask = binary_dilation(
            bridges > 0, iterations=BRIDGE_VISUAL_THICKNESS)

        c_bridge = mcolors.to_rgba(COLOR_BRIDGE)
        rgb_map[thick_bridges_mask] = c_bridge

    road_indices = roads > 0
    if np.any(road_indices):
        c_road = mcolors.to_rgba(COLOR_ROAD)
        rgb_map[road_indices] = c_road

    city_indices = cities > 0
    if np.any(city_indices):
        rgb_map[city_indices] = blend_layers(
            rgb_map[city_indices], COLOR_CITY, CITY_ALPHA
        )

    fig, ax = plt.subplots(figsize=(16, 16))

    ax.imshow(rgb_map)
    ax.axis('off')
    ax.set_title(
        "Procedural Fortnite-like Island", fontsize=20, pad=20)

    legend_elements = [
        Patch(facecolor=named_colors['Grasslands'], label='Grasslands'),
        Patch(facecolor=named_colors['Forest'], label='Forest'),
        Patch(facecolor=named_colors['Desert'], label='Desert'),
        Patch(facecolor=named_colors['Snow'], label='Snow'),
        Patch(facecolor=named_colors['Swamp'], label='Swamp'),
        Patch(facecolor=named_colors['Boreal Tundra'], label='Tundra'),
        Patch(facecolor=COLOR_ROAD, label='Roads'),
        Patch(facecolor=COLOR_BRIDGE, label='Bridges'),
        Patch(facecolor=COLOR_CITY, label='Cities'),
        Patch(facecolor=COLOR_TREE, label='Vegetation'),
    ]

    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=12, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    render_map()
