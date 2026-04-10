# ggplot2_py

AI-assisted Python port of the R **ggplot2** package — Create Elegant Data Visualisations Using the Grammar of Graphics.

## Overview

ggplot2_py implements the grammar of graphics in Python, faithfully porting R's ggplot2 using pandas DataFrames as the data container and a Cairo-based rendering backend. It supports 40+ geoms, 30+ stats, faceting, coordinate systems, themes, and scales.

## Dependencies

This package depends on three companion R-to-Python ports from the same project:

| Package | R source | Python import | Repository |
|---------|----------|---------------|------------|
| **grid_py** | `grid` | `grid_py` | [R2pyBioinformatics/grid_py](https://github.com/R2pyBioinformatics/grid_py) |
| **gtable_py** | `gtable` | `gtable_py` | [R2pyBioinformatics/gtable_py](https://github.com/R2pyBioinformatics/gtable_py) |
| **scales_py** | `scales` | `scales` | [R2pyBioinformatics/scales_py](https://github.com/R2pyBioinformatics/scales_py) |

Additional Python dependencies: numpy, pandas, matplotlib, scipy, pycairo.

## Installation

```bash
# Install companion packages first
pip install git+https://github.com/R2pyBioinformatics/grid_py.git
pip install git+https://github.com/R2pyBioinformatics/gtable_py.git
pip install git+https://github.com/R2pyBioinformatics/scales_py.git

# Install ggplot2_py
pip install git+https://github.com/R2pyBioinformatics/ggplot2_py.git

# Or for development (editable install)
git clone https://github.com/R2pyBioinformatics/ggplot2_py.git
cd ggplot2_py
pip install -e ".[dev]"
```

## Quick Start

```python
from ggplot2_py import *
from ggplot2_py.datasets import mpg

(ggplot(mpg, aes(x="displ", y="hwy", colour="class"))
 + geom_point()
 + geom_smooth(method="lm")
 + facet_wrap("drv")
 + theme_minimal()
 + labs(title="Engine Displacement vs Highway MPG"))
```

## Tutorials

- [Getting Started](tutorials/ggplot2.ipynb) — core concepts: data, aes, geoms, stats, scales, facets, coords, themes
- [Geom Gallery](tutorials/geoms_gallery.ipynb) — boxplot, violin, density, tile, hex and combinations
- [Labels & Facets](tutorials/labels_and_facets.ipynb) — axis titles, plot title/subtitle/caption, facet strip labels
- [Aesthetic Specs](tutorials/aesthetic_specs.ipynb) — colour, fill, alpha, linetype, shape, size, colour scales
- [Extending ggplot2](tutorials/extending_ggplot2.ipynb) — custom stats, geoms, themes via ggproto

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```
