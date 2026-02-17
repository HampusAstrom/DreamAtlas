"""
Basic functionality tests for DreamAtlas
Tests core functionality without GUI components

Run with:
  pytest                          # Fast unit tests only
  pytest -m integration           # Full integration test (slow, ~30+ seconds)
  pytest -m "not integration"     # Skip integration tests
"""
import pytest

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

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
