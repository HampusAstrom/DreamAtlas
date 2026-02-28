import pytest
import numpy as np
from DreamAtlas.classes.class_region import Region
from DreamAtlas.classes.class_settings import DreamAtlasSettings

def test_region_basic_init():
    settings = DreamAtlasSettings(index=0)
    region = Region(index=1, settings=settings, seed=42)
    assert region.index == 1
    assert region.settings is settings
    assert region.seed == 42
    assert region.region_size == 1
    assert region.anchor_connections == 1
    assert region.name == 'testname'
    assert region.plane == 1

def test_region_generate_graph_and_terrain():
    settings = DreamAtlasSettings(index=0)
    region = Region(index=1, settings=settings, seed=42)
    region.region_size = 3
    region.anchor_connections = 2
    region.generate_graph(seed=123)
    assert len(region.provinces) == 3
    from DreamAtlas.databases.dreamatlas_data import TERRAIN_PREF_BITS
    region.terrain_pref = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    region.layout = (0, 1.0, 0.5)
    region.generate_terrain(seed=123)
    for province in region.provinces:
        assert hasattr(province, 'terrain_int')

def test_region_generate_population():
    settings = DreamAtlasSettings(index=0)
    settings.pop_balancing = 1
    region = Region(index=1, settings=settings, seed=42)
    region.region_size = 2
    region.anchor_connections = 1
    region.generate_graph(seed=123)
    from DreamAtlas.databases.dreamatlas_data import TERRAIN_PREF_BITS
    region.terrain_pref = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    region.layout = (0, 1.0, 0.5)
    region.generate_terrain(seed=123)
    region.generate_population(seed=123)
    for province in region.provinces:
        assert province.population is not None

def test_region_embed_region():
    settings = DreamAtlasSettings(index=0)
    region = Region(index=1, settings=settings, seed=42)
    region.region_size = 2
    region.anchor_connections = 1
    region.generate_graph(seed=123)
    from DreamAtlas.databases.dreamatlas_data import TERRAIN_PREF_BITS
    region.terrain_pref = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    region.layout = (0, 1.0, 0.5)
    region.generate_terrain(seed=123)
    region.generate_population(seed=123)
    global_coordinates = [10, 10]
    scale = [(1, 1), (1, 1), (1, 1)]
    map_size = [(100, 100), (100, 100), (100, 100)]
    region.embed_region(global_coordinates, scale, map_size, seed=123)
    for province in region.provinces:
        assert hasattr(province, 'coordinates')

def test_region_str():
    settings = DreamAtlasSettings(index=0)
    region = Region(index=1, settings=settings, seed=42)
    s = str(region)
    assert 'Region' in s or 'type' in s
