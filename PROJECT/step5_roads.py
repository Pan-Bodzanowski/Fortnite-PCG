import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import center_of_mass, binary_dilation, gaussian_gradient_magnitude
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev
import heapq
from tqdm import tqdm
from pathlib import Path

# folders
DATA_DIR = Path('data')
IMAGES_DIR = Path('images')
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TERRAIN_FILE = DATA_DIR / 'step4_terrain_with_cities.npy'
CITIES_MASK_FILE = DATA_DIR / 'step4_cities_mask.npy'
OUTPUT_ROADS_MASK = DATA_DIR / 'step5_roads_mask.npy'
OUTPUT_IMAGE = IMAGES_DIR / 'step5_road_network.png'

SQRT_2 = 1.4142
INF_COST = 1e12


def heuristic(a, b):
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    dist = max(dx, dy) + (SQRT_2 - 1) * min(dx, dy)
    return dist


def generate_cost_map(terrain, sea_level, cost_flat, cost_water, slope_weight, slope_power):
    grad = gaussian_gradient_magnitude(terrain, sigma=1.5)
    slope_cost = np.power(grad * slope_weight, slope_power)
    is_water = (terrain <= sea_level).astype(float)
    water_cost = is_water * cost_water
    return cost_flat + slope_cost + water_cost


def get_move_cost(current_pos, next_pos, terrain, static_cost_map, roads_mask, max_slope_diff, cost_existing_road):
    r2, c2 = next_pos
    H, W = terrain.shape

    if not (0 <= r2 < H and 0 <= c2 < W):
        return INF_COST

    if roads_mask[r2, c2] > 0:
        return cost_existing_road

    h1 = terrain[current_pos]
    h2 = terrain[next_pos]
    if abs(h2 - h1) > max_slope_diff:
        return INF_COST

    return static_cost_map[r2, c2]


def astar_path(start, goal, terrain, static_cost_map, roads_mask, max_slope_diff, cost_existing_road):
    H, W = terrain.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    pq = []
    heapq.heappush(pq, (0, 0, start))

    came_from = {start: None}
    cost_so_far = {start: 0}

    final_node = None
    max_iter = H * W * 3
    iter_count = 0

    neighbors_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while pq:
        iter_count += 1
        if iter_count > max_iter:
            break

        _, current_cost, current = heapq.heappop(pq)

        if current == goal:
            final_node = current
            break

        for dr, dc in neighbors_deltas:
            nr, nc = current[0] + dr, current[1] + dc
            next_node = (nr, nc)

            move_cost = get_move_cost(
                current, next_node, terrain, static_cost_map, roads_mask,
                max_slope_diff, cost_existing_road
            )

            if move_cost >= INF_COST:
                continue

            dist_mult = SQRT_2 if dr != 0 and dc != 0 else 1.0
            new_cost = current_cost + (move_cost * dist_mult)

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal)
                heapq.heappush(pq, (priority, new_cost, next_node))
                came_from[next_node] = current

    if final_node is None:
        return None

    path = []
    curr = final_node
    while curr is not None and curr != start:
        path.append(curr)
        curr = came_from.get(curr)
    path.append(start)
    return path[::-1]


def get_smooth_path(path_coords, smoothing=10, step=5):
    if len(path_coords) < 4:
        return path_coords

    y = [p[0] for p in path_coords]
    x = [p[1] for p in path_coords]

    y_sub = [y[0]] + y[1:-1:step] + [y[-1]]
    x_sub = [x[0]] + x[1:-1:step] + [x[-1]]

    if len(x_sub) < 4:
        return path_coords

    try:
        tck, _ = splprep([y_sub, x_sub], s=smoothing, k=3, per=False)
        u_new = np.linspace(0, 1, len(path_coords))
        new_y, new_x = splev(u_new, tck)
        return [(int(ny), int(nx)) for ny, nx in zip(new_y, new_x)]
    except Exception:
        return path_coords


def get_city_centers(cities_mask):
    ids = np.unique(cities_mask)
    ids = ids[ids > 0]
    return [tuple(map(int, center_of_mass(cities_mask == i))) for i in ids]


def build_network(terrain, cities_mask, sea_level,
                  road_thickness=5,
                  cost_flat=1.0,
                  cost_water=1000.0,
                  slope_weight=800.0,
                  slope_power=3,
                  max_slope_diff=0.12,
                  cost_existing_road=0.75):
    H, W = terrain.shape

    print("Generating static cost map...")
    static_cost_map = generate_cost_map(
        terrain, sea_level,
        cost_flat, cost_water, slope_weight, slope_power
    )

    city_centers = get_city_centers(cities_mask)
    if len(city_centers) < 3:
        return np.zeros_like(terrain)

    print("Computing Delaunay triangulation...")
    tri = Delaunay(np.array(city_centers))
    edges = set()
    for simplex in tri.simplices:
        p1, p2, p3 = sorted(simplex)
        edges.add((p1, p2))
        edges.add((p2, p3))
        edges.add((p1, p3))

    sorted_edges = []
    max_dist = min(H, W) * 0.8
    for idx1, idx2 in edges:
        p1 = city_centers[idx1]
        p2 = city_centers[idx2]
        d = heuristic(p1, p2)
        if d <= max_dist:
            sorted_edges.append((d, p1, p2))

    sorted_edges.sort(key=lambda x: x[0])

    roads_mask = np.zeros_like(terrain, dtype=int)

    for (_, start, end) in tqdm(sorted_edges, desc="Routing", unit="road"):
        path = astar_path(
            start, end, terrain, static_cost_map, roads_mask,
            max_slope_diff, cost_existing_road
        )

        if path:
            smooth_path = get_smooth_path(path, smoothing=5)
            for r, c in smooth_path:
                if 0 <= r < H and 0 <= c < W:
                    roads_mask[r, c] = 1

    radius = road_thickness // 2
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    brush = x**2 + y**2 <= radius**2
    thick_roads_mask = binary_dilation(roads_mask, structure=brush).astype(int)

    return thick_roads_mask


def visualize_roads(terrain, roads_mask, cities_mask, sea_level, save_path):
    H, W = terrain.shape
    _, ax = plt.subplots(figsize=(12, 12))

    colors = [(0.0, "darkblue"), (sea_level, "blue"), (sea_level, "khaki"),
              (sea_level + 0.1, "forestgreen"), (0.8, "saddlebrown"), (1.0, "white")]
    cmap_terrain = mcolors.LinearSegmentedColormap.from_list("island", colors)

    ax.imshow(terrain, cmap=cmap_terrain,
              interpolation='bilinear', vmin=0, vmax=1)

    overlay_roads = np.zeros((H, W, 4))
    overlay_roads[roads_mask > 0] = [0.96, 0.87, 0.70, 1.0]
    ax.imshow(overlay_roads)

    overlay_cities = np.zeros((H, W, 4))
    overlay_cities[cities_mask > 0] = [0.8, 0.2, 0.2, 0.7]
    ax.imshow(overlay_cities)

    ax.set_title('Organic Road Network', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Map saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    terrain = np.load(TERRAIN_FILE)
    cities_mask = np.load(CITIES_MASK_FILE)

    roads_mask = build_network(
        terrain,
        cities_mask,
        sea_level=0.52
    )

    np.save(OUTPUT_ROADS_MASK, roads_mask)
    visualize_roads(terrain, roads_mask, cities_mask, 0.52, OUTPUT_IMAGE)
