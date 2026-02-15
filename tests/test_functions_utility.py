"""
Unit tests for functions/_minor_functions.py

Tests utility functions that are used throughout the generation pipeline.
These are pure functions, making them easy to test deterministically.
"""
import pytest
from DreamAtlas.functions._minor_functions import (
    terrain_int2list,
    has_terrain,
    find_shape_size
)
from DreamAtlas.classes import DreamAtlasSettings, Province


class TestTerrainInt2List:
    """Test terrain_int2list() - converts bitwise terrain int to list"""

    def test_terrain_int2list_empty(self):
        """terrain_int=0 should return empty list"""
        result = terrain_int2list(0)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_terrain_int2list_single_terrain(self):
        """Single terrain bit should return list with one element"""
        # Terrain 1 = bit 2^0
        result = terrain_int2list(1)

        assert isinstance(result, list)
        assert len(result) == 1
        assert 1 in result

    def test_terrain_int2list_multiple_terrains(self):
        """Multiple terrain bits should return all set bits"""
        # Terrain int: 1 + 2 + 8 = 11 (bits: 2^0, 2^1, 2^3)
        result = terrain_int2list(11)

        assert isinstance(result, list)
        assert 1 in result   # 2^0
        assert 2 in result   # 2^1
        assert 8 in result   # 2^3
        assert len(result) == 3

    def test_terrain_int2list_large_terrain(self):
        """Large terrain int with many bits"""
        # Example: large int with multiple bits set
        terrain_int = 255  # All 8 bits set
        result = terrain_int2list(terrain_int)

        assert len(result) == 8
        expected = [1, 2, 4, 8, 16, 32, 64, 128]
        assert set(result) == set(expected)

    def test_terrain_int2list_preserves_power_of_two(self):
        """Result should only contain powers of two"""
        terrain_int = 2047  # Many bits
        result = terrain_int2list(terrain_int)

        for val in result:
            # Check if it's a power of 2: (val & (val-1)) == 0
            assert (val & (val - 1)) == 0, f"{val} is not a power of 2"


class TestHasTerrain:
    """Test has_terrain() - checks if terrain int contains specific terrain"""

    def test_has_terrain_simple_match(self):
        """has_terrain() should return True for set bit"""
        terrain_int = 5  # Binary: 101 (bits 0 and 2 set)

        assert has_terrain(terrain_int, 1)  # Bit 0 is set
        assert has_terrain(terrain_int, 4)  # Bit 2 is set

    def test_has_terrain_no_match(self):
        """has_terrain() should return False for unset bit"""
        terrain_int = 5  # Binary: 101

        assert not has_terrain(terrain_int, 2)   # Bit 1 is not set
        assert not has_terrain(terrain_int, 8)   # Bit 3 is not set

    def test_has_terrain_zero(self):
        """has_terrain() with terrain_int=0 should always be False"""
        for terrain in [1, 2, 4, 8, 16]:
            assert not has_terrain(0, terrain)

    def test_has_terrain_composite_check(self):
        """has_terrain() should work with composite terrain values"""
        terrain_int = 15  # Binary: 1111 (first 4 bits set)

        # Check combination
        composite = 3  # Binary: 11 (bits 0 and 1)
        assert has_terrain(terrain_int, composite)

    def test_has_terrain_large_values(self):
        """has_terrain() should work with large terrain ints"""
        terrain_int = 2**30 + 2**20  # Very large value

        assert has_terrain(terrain_int, 2**30)
        assert has_terrain(terrain_int, 2**20)
        assert not has_terrain(terrain_int, 2**10)


class TestFindShapeSize:
    """Test find_shape_size() - calculates province size and shape"""

    def test_find_shape_size_returns_tuple(self):
        """find_shape_size() should return (size, shape) tuple"""
        province = Province(index=1)
        province.population = 1000  # Must initialize population
        settings = DreamAtlasSettings(index=0)

        result = find_shape_size(province, settings)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_find_shape_size_values_are_numeric(self):
        """size and shape should be numeric"""
        province = Province(index=1)
        province.population = 1000  # Must initialize population
        settings = DreamAtlasSettings(index=0)

        size, shape = find_shape_size(province, settings)

        assert isinstance(size, (int, float))
        assert isinstance(shape, (int, float))

    def test_find_shape_size_default_province(self):
        """Default province should have valid size and shape"""
        province = Province(index=1)
        settings = DreamAtlasSettings(index=0)
        settings.pop_balancing = 0  # Disable pop balancing for deterministic result

        size, shape = find_shape_size(province, settings)

        # Default shape should be 2 (baseline)
        assert shape >= 1
        assert size > 0

    def test_find_shape_size_with_population(self):
        """Size should increase with population (if pop_balancing enabled)"""
        settings = DreamAtlasSettings(index=0)
        settings.pop_balancing = 1  # Enable pop balancing

        # Province with low population
        province_low = Province(index=1)
        province_low.population = 100
        size_low, _ = find_shape_size(province_low, settings)

        # Province with high population
        province_high = Province(index=2)
        province_high.population = 10000
        size_high, _ = find_shape_size(province_high, settings)

        # Higher population should give higher size
        assert size_high >= size_low

    def test_find_shape_size_consistency(self):
        """find_shape_size() should be deterministic for same inputs"""
        province = Province(index=1)
        province.terrain_int = 16  # Some terrain
        province.population = 1000

        settings = DreamAtlasSettings(index=0)
        settings.pop_balancing = 1

        # Call twice
        result1 = find_shape_size(province, settings)
        result2 = find_shape_size(province, settings)

        # Should get same result
        assert result1 == result2


class TestTerrainFunctionIntegration:
    """Test that terrain functions work together correctly"""

    def test_terrain_int2list_then_has_terrain(self):
        """terrain_int2list() result should all pass has_terrain()"""
        terrain_int = 255  # Multiple bits set
        terrain_list = terrain_int2list(terrain_int)

        for terrain in terrain_list:
            assert has_terrain(terrain_int, terrain)

    def test_empty_terrain_int_consistency(self):
        """Empty terrain int should behave consistently"""
        terrain_int = 0

        # terrain_int2list should return empty
        assert len(terrain_int2list(terrain_int)) == 0

        # has_terrain should always be False
        for test_terrain in [1, 2, 4, 8, 16]:
            assert not has_terrain(terrain_int, test_terrain)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
