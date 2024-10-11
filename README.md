# Geospatial Tools

[![License](https://img.shields.io/github/license/geoparse/geospatial)](https://github.com/geoparse/geospatial/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Contributors](https://img.shields.io/github/contributors/geoparse/geospatial)](https://github.com/geoparse/geospatial/graphs/contributors)

A collection of Python functions for performing geospatial analysis and creating visualisations.

A Python library for geospatial data processing, map matching, and OSM-based geometry handling. This library provides efficient geospatial indexing, geometry manipulations, and various utilities to work with OpenStreetMap data.

## Features

- Convert OpenStreetMap (OSM) way IDs to geometries.
- Decode polylines to latitude/longitude coordinates.
- Perform map-matching using different services (e.g., OSRM).
- Utilities for working with OSM geometries.
- Fast intersection methods for spatial data analysis.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Convert OSM way ID to geometry](#convert-osm-way-id-to-geometry)
  - [Decode polyline](#decode-polyline)
  - [Map matching](#map-matching)
  - [Intersection utilities](#intersection-utilities)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/geoparse/geospatial.git

# gutils.py - Geospatial Utility Functions

gutils.py is a collection of geospatial utility functions designed to simplify common geospatial data processing tasks. These functions provide capabilities for geometry statistics, coordinate transformations, spatial intersections, geocoding, and more.



# gindex.py - Geospatial Indexing Tools

This repository provides a set of tools for efficient geospatial indexing and spatial analysis using widely adopted spatial indexing systems such as **Geohash**, **S2**, and **H3**. These tools allow converting geometries into spatial cells, compacting/uncompacting cells, and performing various spatial operations, optimized for performance with parallel processing capabilities.

## Features

- **Convert Geometries to Spatial Cells**: Easily convert geometries (e.g., Polygons, MultiPolygons) to spatial cells in formats like Geohash, S2, or H3.
- **Parallelized Processing**: Process large datasets in parallel to speed up cell generation from geometries.
- **Cell Operations**: Compact, uncompact, and convert spatial cells back to their geometrical form.
- **Customizable Resolutions**: Define the resolution level to control the granularity of spatial cells.
- **Spatial Statistics**: Compute H3 cell statistics like cell coverage and area for given geometries.

## Supported Indexing Systems

- **Geohash**: Divides the Earth's surface into a grid of hierarchical geohashes.
- **S2**: A spherical geometry system that partitions the globe into hierarchical cells.
- **H3**: A hexagonal grid system used for efficient spatial operations.

# karta.py - Visualization Tools

Karta means map in Swedish and `karta.py` module generates visual representations of spatial data to assist in analysis and interpretation.

## Usage

To use the functions in `karta.py`, you can import the module into your Python script or interactive environment. Here’s a brief example of how to get started:

# Example of a visualisation
result = some_function(parameters)  # Call the function with appropriate parameters

