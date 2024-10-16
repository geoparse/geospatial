# GeoParse

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![License](https://img.shields.io/github/license/geoparse/geospatial)](https://github.com/geoparse/geospatial/blob/main/LICENSE)
[![PythonVersion]( https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Contributors](https://img.shields.io/github/contributors/geoparse/geospatial)](https://github.com/geoparse/geospatial/graphs/contributors)



GeoParse is a Python library designed for geospatial data analysis and visualization. It builds on top of popular libraries like GeoPandas and Folium, providing a powerful toolkit for working with geospatial data. GeoParse focuses on efficient geospatial indexing, geometry manipulations, and utilities to handle OpenStreetMap data with ease.



### Key Features
---
* Efficient geospatial indexing using grid-based systems (Geohash, H3, S2)
* Geometry manipulations and conversions between formats
* Utilities for working with OpenStreetMap (OSM) data
* Data visualization using Folium maps

Visit the official GeoParse homepage for more details.

### Quick Start
---
You can run Geoparse examples directly in your browser—no installation required! 
Use the following links to access interactive Jupyter Notebooks hosted on MyBinder:

Latest Stable Version: Binder

### Documentation
---
For a quick introduction, we recommend starting with the tutorial notebooks available on the Geoparse homepage. The official API documentation is hosted on ReadTheDocs: Geoparse API Docs.



# Dependencies used by **geoparse**

All dependencies used by ```geoparse``` are as follows:

```numpy```, ```pandas```, ```geopandas```, ```matplotlib```. By installing ```geoparse``` all these packages will be installed!.

# Installation

There are two options to install Geoparse locally.

## 1. From PyPI

**geoparse** is available on [PyPI](https://pypi.org/project/geo-parse/), so to install it, run this command in your terminal:

`pip install geo-parse`

## 2. Installing from source

It is also possible to install the latest development version directly from the GitHub repository with:

`pip install git+https://github.com/geoparse/geospatial.git`


Feel free to explore the tutorial notebooks and dive into the official documentation to get started with GeoParse!







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


# gutils.py - Geospatial Utility Functions

gutils.py is a collection of geospatial utility functions designed to simplify common geospatial data processing tasks. These functions provide capabilities for geometry statistics, coordinate transformations, spatial intersections, geocoding, and more.




# GIndex: Spatial Cell Indexing for Geospatial Data

GIndex is a Python library for efficiently converting and manipulating geospatial geometries into spatial cell systems like Geohash, S2, and H3. The library provides utilities for parallelized processing, compacting/uncompacting cells, and performing intersection-based slicing to improve performance on large geospatial datasets.
Features

    Spatial cell conversion: Convert geospatial polygons into Geohash, S2, or H3 cells at custom resolution levels.
    Parallelized processing: Utilize multiple CPU cores to handle large GeoDataFrames in parallel.
    Cell compaction: Reduce the number of spatial cells by merging adjacent cells into parent cells, based on supported hierarchies (S2, H3).
    Save to disk: Optionally save generated spatial cells to disk for long-term storage and retrieval.


    


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

