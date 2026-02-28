# DreamAtlas
Dominions 6 map generator

DreamAtlas is an open-source Python library for Dominions 6 map editing and generation. DreamAtlas allows users to seamlessly interact with Dominions maps through Python code. Equipped with groundbreaking map generation algorithms and tools for artists that will allow maps to be made better than ever before.

I've always loved fantasy, and a major part of fantasy for me is elaborate maps, epic mountain ranges, rivers and so on. I also love competition and multiplayer strategy games. Dominions has never really brought this together for me, most maps are either focused on nice geography or good balance (or neither in vanilla), so I decided to make my own map generator to make the kind of maps I'd like to play on. This meant going to the drawing board and really asking what makes a good dominions map and what maths you need to accomplish that? Further I wanted to make a tool not just to make maps, but to make map generators and support people making maps from custom art, as a result the DreamAtlas generator is essentially just a script calling functions and classes from the DreamAtlas library, anyone with some python knowledge could make their own mapmaker with DreamAtlas using the generalised tools, classes and methods it comes with.

## Project Structure

This monorepo contains two packages in the `src/` directory:

- **DreamAtlas** - Main map generation and editing framework
- **FlowAtlas** - Experimental flow-based map generation (experimental alternative generator)

Both packages are installed together and can be imported independently:
```python
from DreamAtlas.classes import DominionsMap, DreamAtlasSettings
from FlowAtlas.flow_types import FlowSettings
```

## Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (or Python 3.8+)
- Git

### Setup with Conda (Recommended)

1. **Create a new conda environment:**
   ```bash
   conda create -n dreamatlas python=3.11
   conda activate dreamatlas
   ```

2. **Clone and navigate to the repository:**
   ```bash
   cd path/to/procedural_generation/DreamAtlas
   ```

3. **Install both DreamAtlas and FlowAtlas in development mode:**
   ```bash
   pip install -e .
   ```
   This installs both packages with all dependencies in editable mode, allowing changes to be reflected immediately.

4. **Verify installation:**
   ```bash
   python -c "from DreamAtlas.classes import DreamAtlasSettings; from FlowAtlas.flow_types import FlowSettings; print('✓ DreamAtlas installed'); print('✓ FlowAtlas installed')"
   ```

### Setup with Python venv

If you prefer not to use conda:

```bash
# Create and activate virtual environment
python -m venv dreamatlas_env
source dreamatlas_env/bin/activate  # On Windows: dreamatlas_env\Scripts\activate

# Install packages
cd path/to/procedural_generation/DreamAtlas
pip install -e .
```

## Usage

### Running the GUI

Launch the interactive DreamAtlas UI:

```bash
python examples/run.py
```

### Running Headless Generation

Generate a map without the GUI:

```bash
python examples/no_gui_run.py
```

You can specify a custom config file:

```bash
python examples/no_gui_run.py --config path/to/config.dream
```

### As a Library

```python
from DreamAtlas.classes import DreamAtlasSettings, DominionsMap
from DreamAtlas.generators import generator_dreamatlas

# Create settings and generate a map
settings = DreamAtlasSettings(index=0)
settings.load_file(filename="path/to/config.dream")
map_obj = generator_dreamatlas(settings=settings)
map_obj.plot()
```

## Testing

Run all tests:

```bash
pytest
```

Run only fast tests (excludes slow integration tests):

```bash
pytest -m "not slow"
```

Run all tests including slow ones:

```bash
pytest -m ""
```

Tests are located in:
- `src/DreamAtlas/tests/` - DreamAtlas tests
- `src/FlowAtlas/tests/` - FlowAtlas tests

## Dependencies

Automatically installed via `setup.py`:
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **matplotlib** - Visualization
- **networkx** - Graph/network operations
- **minorminer** - Graph embedding
- **ttkbootstrap** - Modern themed GUI toolkit
- **noise** - Perlin noise generation
- **numba** - JIT compilation for performance
- **Pillow** - Image processing
- **geopandas** - Geospatial data handling
- **libpysal** - Spatial analysis library

## Development

### Project Layout

```
src/
├── DreamAtlas/          # Main package
│   ├── classes/         # Core classes (Map, Region, etc.)
│   ├── functions/       # Utility functions
│   ├── generators/      # Map generation algorithms
│   ├── GUI/             # GUI components
│   ├── databases/       # Game data and resources
│   └── tests/           # Unit tests
└── FlowAtlas/           # Alternative generator (experimental)
    ├── flow_types.py    # Flow graph types
    ├── map_generator.py # Generation logic
    └── tests/           # FlowAtlas tests
```

### Key Files

- `setup.py` - Package configuration for both DreamAtlas and FlowAtlas
- `pytest.ini` - Test configuration
- `environment.yml` - Conda environment specification
- `conftest.py` - Pytest configuration

## License

[Include license information]

## Contributing

[Include contribution guidelines]
