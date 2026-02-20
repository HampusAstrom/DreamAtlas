# DreamAtlas Code Architecture

## Project Overview

DreamAtlas is a procedural map generator for Dominions, a turn-based fantasy strategy game. It generates 2D fantasy maps with nations, provinces, and terrain using graph-based algorithms and Voronoi relaxation.

**Key Technologies:**
- Python 3.11
- NumPy for numerical operations
- Numba for JIT compilation of performance-critical code
- tkinter + ttkbootstrap for GUI
- NumPy/SciPy for graph algorithms

---

## Directory Structure & Module Overview

### `classes/` - Data Structures
Defines core data containers and objects used throughout generation.

#### `class_map.py` - DominionsMap
Central container for all generated map data.
```python
class DominionsMap:
    settings: DreamAtlasSettings | None  # User settings that generated this map
    image_file: list[str | None]         # Paths to generated image files
    pixel_map: list[np.ndarray | None]   # Region assignment by pixel (2D arrays)
    height_map: list[np.ndarray | None]  # Terrain height data (2D arrays)
    min_dist: list[float | None]         # Minimum inter-region distances
    planes: list[int]                    # Which map planes were generated (2D, 3D, hell, etc.)
    seed: int                            # Random seed used for generation
```
**Key Methods:**
- Initialization with default values
- Attribute access for downstream operations

#### `class_settings.py` - DreamAtlasSettings
User-facing configuration container.
```python
class DreamAtlasSettings:
    map_title: str                       # User-provided map name
    seed: int | None                     # Random seed (-1 for autogenerate)
    nations: list                        # Selected nation indices
    index: int                           # Database configuration index
    # ... province/terrain/population settings
```
**Critical**: Constructor requires `index` parameter (selects database configuration).

#### `class_nation.py` - Nation Classes
Represents player nations with preferences.

```python
class Nation:
    # Database-backed lookups via [index, subindex]
    terrain_profile: list                # Terrain preference weights
    layout: list                         # Layout preference (scale, cluster, something)
    home_plane: int                      # Which plane this nation starts on
    terrain: str | int                   # Terrain type identifier
    iid: int | None                      # Instance ID (set during generation)

class CustomNation(Nation):
    # User-created nation (custom index ranges)
    pass

class GenericNation(Nation):
    # Generic/default nation (no name, index allocation)
    pass
```
**Important Pattern**: Database lookups are lazy - accessed when attributes are read.

#### `class_province.py`, `class_region.py`, `class_layout.py`
Define provinces (player-owned subdivisions) and regions (natural map areas).

#### `graph.py`
Graph representations for map structure, with Numba-optimized operations:
- Attractor adjustment (graph refinement)
- Spring adjustment (force-directed layout)
- Norm calculations

---

### `functions/` - Algorithms & Utilities

#### `_minor_functions.py`
Pure utility functions used throughout generation:
```python
def terrain_int2list(terrain_int: int) -> list[int]:
    # Convert bitmask to terrain list
    # Example: 0b101 → [0, 2]

def has_terrain(terrain_int: int, terrain: int) -> bool:
    # Check if specific terrain bit is set

def find_shape_size(province, settings) -> tuple[float, float]:
    # Calculate province size/shape based on population
```

**Pattern**: These are pure functions (no side effects) with stable signatures.

#### `functions_graph_embedding.py`
Graph layout algorithms (force-directed placement, stress minimization).

#### `functions_lloyd.py`
Lloyd relaxation (Voronoi centroid-based refinement) for smooth region distribution.

#### `numba_pixel_mapping.py`
Numba-compiled algorithms for performance:
- Jump flood algorithm (fast distance calculations)
- Euclidean distance (2D)
- Height map generation
- Pixel-to-region mapping

**Note**: Numba requires pure NumPy operations - no Python objects in compiled code.

---

### `generators/` - Generation Orchestration

#### `DreamAtlas_map_generator.py` - PRIMARY GENERATOR
```python
def generate_dreamatlas(settings: DreamAtlasSettings, seed: int = None) -> DominionsMap:
    """Main entry point for map generation."""
    # Flow:
    # 1. Create DominionsMap container
    # 2. Create DominionsLayout
    # 3. Call dibber() - main algorithm
    # 4. Return populated DominionsMap
```

**Critical Initialization Pattern**:
```python
map_class = DominionsMap()                          # Create container
map_class.map_title = settings.map_title            # Store settings
map_class.settings, map_class.seed = settings, seed
dibber(map_class, seed)                             # Run generation
return map_class                                     # Return populated map
```
**Note**: This sequence was previously deleted and restored as critical bug fix.

#### `DreamAtlas_geo_generator.py`
Geographic generation details (terrain, height maps, etc.).

---

### `GUI/` - User Interface (tkinter + ttkbootstrap)

#### `main_interface.py`
Main window layout and controls.

#### `loading.py` - ThreadedGenerator
Background worker for non-blocking generation:
```python
class ThreadedGenerator(threading.Thread):
    def __init__(self, queue: Queue, ui, settings: DreamAtlasSettings):
        super().__init__()
        self.queue = queue
        self.ui = ui
        self.settings = settings
        self.map = None  # CRITICAL: Initialize to None

    def run(self):
        try:
            # Run generation in background
            self.map = generator_dreamatlas(self.settings)
        except Exception as e:
            # Catch and log exceptions for visibility
            self.queue.put(("EXCEPTION", str(e)))
        finally:
            # Notify main thread
            self.queue.put(("TASK_FINISHED", None))
```

**Pattern**: Queue-based communication prevents thread synchronization bugs.
- Main thread sends: generation requests
- Worker thread sends: "TASK_FINISHED" and exceptions
- Main thread polls via `process_queue()` on timer

#### `widgets.py`
Custom tkinter widgets:
```python
class VanillaNationWidget(ttk.Frame):
    # Nation selection widget
    def __init__(self, master):
        super().__init__(master)
        self.options: dict = {}      # Nation options/labels
        self.variables: dict = {}    # StringVar for selection
        self.teams: dict = {}        # Team assignments
        # ... initialization

class InputWidget(ttk.Frame):
    # Generic input field with validation
    target_class: str         # Type assertion string
    target_location: str | None
    # ... implementation
```

**Critical Fix**: These dicts must be initialized as empty dicts (not None).

---

### `databases/` - Data & Configuration

#### `dreamatlas_data.py`
Core database definitions:
```python
TERRAIN_PREFERENCES = {
    # terrain_profile data by nation index
    11: [5, 1, 1, 1, 4, 2],  # Example: nation 11's terrain preferences
    # ...
}

LAYOUT_PREFERENCES = {
    # layout data by nation index
    11: [1, 1.0, 0.85],      # Example: nation 11's layout preferences
    # ...
}

HOMELANDS_INFO = {
    # Nation metadata indexed as [index][subindex]
    11: {
        1: Nation_object_with_metadata,  # Retrieved by Nation([11, 1])
        # ...
    },
    # ...
}
```

**Usage Pattern**:
```python
# In class_nation.py
nation = Nation([11, 1])  # Looks up HOMELANDS_INFO[11][1]
# Accesses: nation.terrain_profile, nation.layout, nation.home_plane
```

#### `dominions_data.py`
Dominions-specific constants and game rules.

---

## Generation Pipeline & Algorithm

### High-Level Flow
```
1. User Input (GUI)
   ↓
2. DreamAtlasSettings validation
   ↓
3. generate_dreamatlas(settings, seed) starts
   a) Create DominionsMap container
   b) Create DominionsLayout (nation placements)
   c) Call dibber() - main generation
   ↓
4. dibber() sequence:
   a) Generate base graph (Delaunay/proximity graph)
   b) Apply graph algorithms (attractor/spring adjustments)
   c) Lloyd relaxation (smooth region distribution)
   d) Create provinces within regions
   e) Assign terrains to provinces (based on nation preferences)
   f) Place nations (PlayerNation, CustomNation, GenericNation)
   g) Map pixels to regions (Numba jump-flood algorithm)
   h) Generate height maps for each plane
   i) Export to D6M file format
   ↓
5. Return populated DominionsMap with:
   - image_file: Generated image paths
   - pixel_map: Which region each pixel belongs to
   - height_map: Terrain height per pixel
   - settings: Original user settings
   - seed: Seed used
```

### Key Algorithm Components

#### Graph Construction (graph.py)
1. **Delaunay Triangulation**: Creates initial graph connectivity
2. **Attractor Adjustment**: Refines node positions toward region centroids
3. **Spring Adjustment**: Applies spring-like force to balance spacing
4. **Lloyd Relaxation**: Iteratively moves centroids (smooth distribution)

#### Pixel Mapping (numba_pixel_mapping.py)
1. **Jump Flood Algorithm**: O(log n) distance field calculation
2. **Euclidean Distance**: Per-pixel distance to nearest region
3. **Assignment**: Each pixel assigned to closest region
4. **Height Calculation**: Perlin/Simplex noise combined with region topology

#### Terrain Assignment (functions)
1. **Preference Lookup**: nation.terrain_profile - weighted terrain preferences
2. **Distribution**: Allocate terrain types to provinces per preference
3. **Clustering**: Group similar terrains (realistic landscape)

---

## Type System & Design Patterns

### Type Annotation Style
```python
# All Python 3.10+ union syntax
def example(value: int | None = None) -> str | None:
    return value and str(value)

# Class attributes with optional values
class Example:
    data: list[str | None] = None  # Can be None or list of optional strings
    value: int | None              # Can be int or None
```

### Type Narrowing Pattern (CRITICAL)
```python
# CORRECT - Pylance understands type narrowing:
value = optional_value
if value is not None:
    # Pylance knows value is not None here
    return value.upper()

# Also correct - with assertion:
value = optional_value
assert value is not None, "Error message"
# Pylance narrows type after assertion
return value.upper()

# WRONG - Pylance won't narrow across separate statements:
assert optional_value is not None
optional_value.upper()  # ERROR: Pylance doesn't know it's not None
```

### Common Patterns

#### Database Lookups
```python
from databases.dreamatlas_data import TERRAIN_PREFERENCES

nation = Nation([11, 1])
# Internally:
# 1. Look up HOMELANDS_INFO[11][1] → get nation data
# 2. Extract terrain_profile from TERRAIN_PREFERENCES[11]
terrain = nation.terrain_profile  # Returns list[int]
```

#### Settings Initialization
```python
# Settings always need an index
settings = DreamAtlasSettings(index=0)  # Loads config 0

# Common convention:
# index=0 → "12_player_ea_test.dream" configuration
```

#### Exception Handling in Threading
```python
def run(self):
    try:
        self.result = expensive_operation()
    except ValueError as e:
        self.error = str(e)
    except Exception as e:
        self.error = f"Unexpected: {e}"
    finally:
        # Always signal completion
        self.notify_done()
```

---

## Testing Strategy

### Test Categories

#### Unit Tests (database, pure functions)
- Deterministic inputs and outputs
- No external dependencies
- Fast execution (<100ms each)
- Located: `test_functions_utility.py`, `test_classes_nation.py`

#### Regression Tests (GUI, threading)
- Verify specific bug fixes
- Test interaction patterns
- Catch accidental breakage
- Located: `test_gui_loading.py`, `test_gui_widgets.py`

#### Integration Tests (full pipeline)
- End-to-end generation validation
- Real settings and database
- Slow (30+ seconds)
- Located: `test_basic_functionality.py` (marked with `integration`)

### Test Data
- Use actual database configurations (e.g., index=0)
- Use real settings from 12_player_ea_test.dream
- Avoid mocks where possible (test real interactions)

---

## Development Guidelines

### Adding New Features
1. **Plan**: Identify module (classes, functions, or generators)
2. **Implement**: Add code following existing patterns
3. **Test**: Create regression/unit tests for new behavior
4. **Verify**: Run full test suite, ensure GUI still loads
5. **Type Check**: Fix any Pylance errors in modified code

### Debugging Generation Issues
1. **Check settings**: Verify DreamAtlasSettings is initialized correctly
2. **Check database**: Verify nation/terrain data is being looked up
3. **Run tests**: Test individual components in isolation
4. **Add logging**: Print intermediate state in generation function
5. **Use integration test**: Run full pipeline to find where it breaks

### Type Checking Issues
1. **Pylance warnings**: Generally safe to ignore if code works
2. **Use `# type: ignore`**: For GUI attributes that can't be typed
3. **Extract-then-assert**: Always extract and assert, never assert-then-extract
4. **Document limitations**: Add comments explaining un-typeable patterns

---

## Performance Considerations

### Numba Functions
- Must be pure NumPy (no Python objects)
- Careful with type signatures (@njit decorators)
- Significant speedup for pixel mapping (~10-100x)

### Graph Algorithms
- Lloyd relaxation is O(n²) per iteration - iterates ~10-50 times
- Limit graph size for real-time feedback in UI

### Memory Usage
- Height maps and pixel maps are large (proportional to image size)
- Typical 1024×1024 map ≈ 8MB per plane
- Multi-plane maps multiply memory usage

---

## Known Limitations

1. **Type System**: tkinter widgets have dynamic attributes that can't be fully typed
   - Use `# type: ignore` pragmatically
   - Document custom attributes

2. **Optional Types**: Some attributes are lazily initialized
   - Pattern: `value: Type | None` with assertions where used
   - Not all optional attributes are always None

3. **ThreadedGenerator**: Must be subclassed properly to work correctly
   - Requires `super().__init__()` call
   - Must initialize all attributes in `__init__`

4. **Database Lookups**: Inflexible structure (can't easily add nations at runtime)
   - Nation indices are pre-defined in database
   - CustomNation uses different index range

---

## Future Improvements

- [ ] Implement `special_neighbours()` method in class_map.py (TODO)
- [ ] Add more unit tests for complex algorithms (Lloyd, graph embedding)
- [ ] Performance profiling and optimization
- [ ] Better error messages for invalid settings
- [ ] Export format documentation
