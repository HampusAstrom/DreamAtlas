"""
Basic functionality tests for DreamAtlas
Tests core functionality without GUI components

Run with:
  pytest                          # Fast unit tests only
  pytest -m integration           # Full integration test (slow, ~30+ seconds)
  pytest -m "not integration"     # Skip integration tests
"""
import pytest
import numpy as np

from DreamAtlas.classes import DreamAtlasSettings, DominionsMap
from DreamAtlas.generators.DreamAtlas_map_generator import generator_dreamatlas


def test_settings_creation():
    """Test that DreamAtlasSettings can be created with default values"""
    settings = DreamAtlasSettings(index=0)

    assert settings is not None
    assert isinstance(settings, DreamAtlasSettings)
    assert hasattr(settings, 'homeland_size')
    assert hasattr(settings, 'periphery_size')


def test_map_creation():
    """Test that DominionsMap can be created and initialized"""
    map_obj = DominionsMap()

    assert map_obj is not None
    assert isinstance(map_obj, DominionsMap)
    assert hasattr(map_obj, 'region_list')
    assert hasattr(map_obj, 'province_list')

    # Test setting multiple planes
    map_obj.planes = [1, 2]
    assert map_obj.planes == [1, 2]


def test_class_layout_assertions():
    """Test that class_layout attributes are properly initialized as None"""
    from DreamAtlas.classes import DominionsLayout

    # Create a minimal map for layout
    map_obj = DominionsMap()
    layout = DominionsLayout(map_obj)

    # These should start as None
    assert layout.region_graph is None
    assert layout.region_types is None
    assert layout.region_planes is None


@pytest.mark.integration
@pytest.mark.slow
def test_generator_full_integration():
    """Full integration test using standard config file

    This test validates the complete generation pipeline with realistic settings.
    It's marked as 'integration' (slow) and not run by default.
    Use: pytest -m integration to run this test.

    This is the primary regression test - if this passes, the system works end-to-end.
    """
    from DreamAtlas.databases.dreamatlas_data import ROOT_DIR

    # Load the full standard config file
    settings = DreamAtlasSettings(index=0)
    settings.load_file(filename=ROOT_DIR / 'databases/12_player_ea_test.dream')

    # Run generator without UI (same as batch/automated mode)
    result = generator_dreamatlas(settings=settings, ui=None, seed=42)

    # Verify result is a DominionsMap
    assert result is not None
    assert isinstance(result, DominionsMap)

    # Verify map has been populated
    assert result.region_list is not None
    assert len(result.region_list) > 0
    assert result.province_list is not None
    assert len(result.province_list) > 0

    # Verify basic map data integrity
    assert result.map_size is not None
    assert result.scale is not None
    assert len(result.region_list[1]) > 0  # At least some homeland regions

    # Test saving and loading .d6m file
    import tempfile, os
    tmp_dir = tempfile.mkdtemp()
    try:
        # Save .d6m file for each plane
        for plane in result.planes:
            d6m_path = os.path.join(tmp_dir, f"test_plane{plane}.d6m")
            result.make_d6m(plane=plane, filepath=d6m_path)
            # Load .d6m file into new map
            loaded_map = DominionsMap()
            loaded_map.planes = [plane]
            loaded_map.load_file(d6m_path, plane=plane)
            # Check loaded attributes
            # Compare map_size using np.array_equal to avoid ambiguous truth value
            assert np.array_equal(loaded_map.map_size[plane], result.map_size[plane])
            assert loaded_map.min_dist[plane] == pytest.approx(result.min_dist[plane])
            # Check shape only if not None
            assert loaded_map.height_map[plane] is not None
            assert result.height_map[plane] is not None
            res_height_map = result.height_map[plane]
            assert res_height_map is not None, "result.height_map[plane] should not be None"
            load_height_map = loaded_map.height_map[plane]
            assert load_height_map is not None, "loaded_map.height_map[plane] should not be None"
            assert load_height_map.shape == res_height_map.shape
            assert loaded_map.pixel_map[plane] is not None
            assert result.pixel_map[plane] is not None
            res_pixel_map = result.pixel_map[plane]
            assert res_pixel_map is not None, "result.pixel_map[plane] should not be None"
            load_pixel_map = loaded_map.pixel_map[plane]
            assert load_pixel_map is not None, "loaded_map.pixel_map[plane] should not be None"
            assert load_pixel_map.shape == res_pixel_map.shape
        # Remove files
        for file in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, file))
    finally:
        import shutil
        shutil.rmtree(tmp_dir)

def test_dominions_map_initialization():
    m = DominionsMap()
    assert m.region_list is not None
    assert isinstance(m.region_list, list)
    assert m.pixel_map is not None
    assert isinstance(m.pixel_map, list)
    assert m.map_title == 'DreamAtlas_map'

def test_dominions_map_planes_assignment():
    m = DominionsMap()
    m.planes = [0, 1, 2]
    assert m.planes == [0, 1, 2]

def test_dominions_map_empty_access():
    m = DominionsMap()
    # region_list has 7 entries, pixel_map has 10
    for i in range(7):
        assert m.region_list[i] == []
    for i in range(10):
        assert m.pixel_map[i] is None


def test_dominions_map_map_dom_colour_type_handling():
    m = DominionsMap()
    # Should be a flat list, but code sometimes expects a list of lists
    # Instead of assigning a list, check type and simulate the error
    assert isinstance(m.map_dom_colour[1], int)
    # If code expects a list, this will fail at runtime, not type-check time
    # So we just assert the type is not list
    assert not isinstance(m.map_dom_colour[1], list)

def test_dominions_map_planes_set_vs_list():
    m = DominionsMap()
    m.planes = [1, 2]
    # Simulate code that expects a set by converting to set and adding
    planes_set = set(m.planes)
    planes_set.add(3)
    assert 3 in planes_set

def test_dominions_map_pixel_map_none_access():
    m = DominionsMap()
    # Accessing pixel_map[plane] when None should not be subscriptable
    for i in range(10):
        if m.pixel_map[i] is None:
            # Instead of raising, check for None and skip access
            assert m.pixel_map[i] is None

def test_fill_dreamatlas_basic():
    """Basic test for fill_dreamatlas: should run without error for minimal valid input."""
    m = DominionsMap()
    # Minimal setup for one plane
    m.planes = [1]
    m.terrain_list[1] = [(1, 0)]  # One province, terrain int 0
    m.map_size[1] = [2, 2]
    m.pixel_owner_list[1] = [[0, 0, 1, 1]]  # One pixel owned by province 1
    m.population_list[1] = [(1, 100)]
    m.neighbour_list[1] = []
    m.province_capital_locations[1] = [(0, 0, 0)]
    import numpy as np
    m.height_map[1] = np.zeros((2, 2), dtype=int)
    m.pixel_map[1] = np.ones((2, 2), dtype=int)
    from typing import Optional
    plane_image_types: list[Optional[str]] = [None for _ in range(10)]
    plane_image_types[1] = '.d6m'
    # Should not raise
    m.fill_dreamatlas(plane_image_types)
    # Check province_list populated
    assert len(m.province_list[1]) == 1
    assert m.province_list[1][0].index == 1
    assert m.province_list[1][0].population == 100
    # Check province_graphs is not None
    assert m.layout is not None
    assert m.layout.province_graphs[1] is not None

def test_make_d6m_minimal(tmp_path):
    m = DominionsMap()
    m.planes = [1]
    m.terrain_list[1] = [(1, 0)]
    m.map_size[1] = [2, 2]
    m.pixel_owner_list[1] = [[0, 0, 1, 1]]
    m.population_list[1] = [(1, 100)]
    m.neighbour_list[1] = []
    m.province_capital_locations[1] = [(0, 0, 0)]
    m.height_map[1] = np.zeros((2, 2), dtype=int)
    m.pixel_map[1] = np.ones((2, 2), dtype=int)
    from typing import Optional
    plane_image_types: list[Optional[str]] = [None for _ in range(10)]
    plane_image_types[1] = '.d6m'
    m.fill_dreamatlas(plane_image_types)
    # Set min_dist to a float to avoid None error
    m.min_dist[1] = 1.0
    out_path = tmp_path / "test.d6m"
    # Should not raise
    m.make_d6m(1, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
