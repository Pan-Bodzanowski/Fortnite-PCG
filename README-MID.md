# Fortnite-PCG

## Abstract

This project focuses on the development of a PCG system designed to automatically create nature-like and balanced Battle Royale mode maps. 
The system focuses on generating a 2D topographical blueprint that serves as the foundation for environmental layout, biome partitioning, and structural placement.

## Background

Procedural Content Generation (PCG) enables the automatic creation of large game environments, reducing manual design effort while increasing replayability. In Battle Royale games, terrain layout strongly influences gameplay balance and player strategy, making natural-looking landscapes especially important. This project applies PCG techniques to generate Fortnite-inspired island maps that imitate real-world terrain features such as varied elevation and coastlines.

## Project goals

We wanted to automatically create maps in 2D (or 3D - ambitious goal), depending on given seed.

## Methodology

The map generation process is divided into several distinct stages, combining procedural noise generation with machine learning for natural-looking results:

### Already completed:
* **Island morphology generation** – a heightmap is generated to define the main island shape, surrounding archipelago, and elevation features. This stage defines the foundational shape and elevation of the terrain using a combination of noise and distance functions. Our script utilizes Perlin noise layered across multiple octaves to create fractal noise. This process uses parameters like persistence and lacunarity to control the ruggedness and detail of the terrain. To ensure the land forms an island and does not touch the map edges, a distance gradient is applied using a power metric (∣x∣p+∣y∣p). This forces the noise values to decrease as they approach the boundaries. The final heightmap is normalized and saved as island_noise_map.npy.

* **Biome partitioning** – To achieve natural and smooth transitions, the island is divided into distinct biomes using K-means clustering that analyzes pixel coordinates alongside heightmap data. By adjusting the height_weight parameter, the algorithm balances whether biomes follow elevation contours like altitudinal zones or form irregular geographic patches across the landscape. Finally, the clusters are deterministically ranked by mean elevation to ensure that biome IDs consistently represent the terrain's progression from sea level to mountain peaks.

### In progress:

* **Terrain decoration** – biome-specific environmental elements (for example canyons in desert biomes or higher mountains in mountain biomes).
    * **Morphology-Integrated Features** (In Progress)

        Major geological structures are generated alongside the base heightmap so that the clustering algorithm can account for them when defining natural biome boundaries. High-elevation mountains are already implemented using a custom layer that places Gaussian-like masks at random coordinates and fills them with ridge-like fractal noise to simulate sharp, realistic peaks. By incorporating these peaks into the initial morphology, the K-means algorithm effectively identifies them as distinct high-altitude biomes based on their elevation values.

    * **Post-Partition Decoration** (To be done)

        Once the biome map is finalized, specific environmental elements like canyons in deserts or dense forests in temperate zones are added to the terrain. This stage utilizes rule-based placement and spatial constraints to prevent overlapping objects and ensure a logical distribution across the map. Every element is placed to remain consistent with its specific climate and the local terrain slope to maintain visual coherence.

### To be done:

* **Cities placement and path generation** – cities are procedurally placed and connected with efficient traversal paths using search-based and heuristic methods (Simulated Annealing).

## Bibliography/References
* UWr Course: Artificial Intelligence <3 Games: Procedural Content Generation slides by Jakub Kowalski (aCat)
* Fortnite Wiki