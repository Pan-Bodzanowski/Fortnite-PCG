# Fortnite-PCG

## Abstract

This project focuses on the development of a PCG system designed to automatically create nature-like and balanced Battle Royale mode maps. 
The system focuses on generating a 2D topographical blueprint that serves as the foundation for environmental layout, biome partitioning, and structural placement.

## Background

Procedural Content Generation (PCG) enables the automatic creation of large game environments, reducing manual design effort while increasing replayability. In Battle Royale games, terrain layout strongly influences gameplay balance and player strategy, making natural-looking landscapes especially important. This project applies PCG techniques to generate Fortnite-inspired island maps that imitate real-world terrain features such as varied elevation and coastlines.

## Project goals & Methodology

We wanted to automatically create maps in 2D (or 3D - ambitious goal), depending on given seed.

The map generation process will be divided into the following stages:
* Island morphology generation – a heightmap is generated to define the main island shape, surrounding archipelago, and elevation features (Perlin Noise).
* Biome partitioning – the island is divided into distinct biomes with smooth and natural transitions using clustering techniques (K-means clustering).
* Terrain decoration – biome-specific environmental elements (for example canyons in desert biomes or higher mountains in mountain biomes) are added while preventing overlaps and ensuring consistency (rule-based placement, spatial constraints).
* Cities placement and path generation – cities are procedurally placed and connected with efficient traversal paths using search-based and heuristic methods (Simulated Annealing).

## Bibliography/References
* UWr Course: Artificial Intelligence <3 Games: Procedural Content Generation slides by Jakub Kowalski (aCat)
* Fortnite Wiki