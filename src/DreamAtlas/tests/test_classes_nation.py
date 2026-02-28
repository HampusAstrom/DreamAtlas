"""
Unit tests for classes/class_nation.py

Tests Nation, CustomNation, and GenericNation creation and attribute initialization.
These are critical for the generation pipeline since nation data drives map generation.
"""
import pytest
from DreamAtlas.classes.class_nation import Nation, CustomNation, GenericNation
from DreamAtlas.databases.dominions_data import ALL_NATIONS
from DreamAtlas.databases.dreamatlas_data import HOMELANDS_INFO, TERRAIN_PREFERENCES, LAYOUT_PREFERENCES


class TestNationCreation:
    """Test Nation class which loads data from ALL_NATIONS database"""

    def test_nation_creation_with_valid_index(self):
        """Nation should create successfully with valid nation index"""
        # Nation 11 is Ulm (exists in ALL_NATIONS)
        nation = Nation([11, 1])

        assert nation is not None
        assert nation.index == 11
        assert nation.team == 1

    def test_nation_name_lookup(self):
        """Nation should look up name from ALL_NATIONS"""
        nation = Nation([11, 1])

        assert hasattr(nation, 'name')
        assert isinstance(nation.name, str)
        assert len(nation.name) > 0

    def test_nation_tagline_lookup(self):
        """Nation should look up tagline from ALL_NATIONS"""
        nation = Nation([11, 1])

        assert hasattr(nation, 'tagline')
        assert isinstance(nation.tagline, str)

    def test_nation_home_plane_lookup(self):
        """Nation should look up home_plane from HOMELANDS_INFO"""
        nation = Nation([11, 1])

        assert hasattr(nation, 'home_plane')
        assert isinstance(nation.home_plane, int)
        assert nation.home_plane in [1, 2]  # Should be surface or underground

    def test_nation_terrain_profile_lookup(self):
        """Nation should look up terrain_profile from HOMELANDS_INFO"""
        nation = Nation([11, 1])

        assert hasattr(nation, 'terrain_profile')
        # terrain_profile is a list from TERRAIN_PREFERENCES
        assert isinstance(nation.terrain_profile, list)

    def test_nation_layout_lookup(self):
        """Nation should look up layout from HOMELANDS_INFO"""
        nation = Nation([11, 1])

        assert hasattr(nation, 'layout')
        # layout is a list from LAYOUT_PREFERENCES
        assert isinstance(nation.layout, list)

    def test_nation_terrain_attribute(self):
        """Nation should have terrain attribute from database"""
        nation = Nation([11, 1])

        assert hasattr(nation, 'terrain')
        assert isinstance(nation.terrain, (int, str))

    def test_nation_iid_initialized_none(self):
        """Nation.iid should start as None (set by generator)"""
        nation = Nation([11, 1])

        assert hasattr(nation, 'iid')
        assert nation.iid is None

    def test_nation_multiple_nations(self):
        """Can create multiple different nation instances"""
        nation1 = Nation([11, 1])
        nation2 = Nation([31, 2])

        assert nation1.index != nation2.index
        assert nation1.team != nation2.team
        # Names should be different
        assert nation1.name != nation2.name

    def test_nation_str_representation(self):
        """Nation.__str__() should work and include attributes"""
        nation = Nation([11, 1])

        str_repr = str(nation)
        assert 'index' in str_repr
        assert 'team' in str_repr
        assert 'name' in str_repr


class TestCustomNationCreation:
    """Test CustomNation class for custom nation creation"""

    def test_custom_nation_creation(self):
        """CustomNation should create from nation data tuple"""
        # Format: [index, name, tagline, terrain, terrain_index, layout_index, home_plane, team]
        nation_data = [600, 'TestNation', 'Test Tagline', 16, 0, 0, 1, 1]
        custom = CustomNation(nation_data)

        assert custom is not None
        assert custom.index == 600
        assert custom.name == 'TestNation'

    def test_custom_nation_attributes(self):
        """CustomNation should have all expected attributes"""
        nation_data = [600, 'TestNation', 'Test Tagline', 16, 0, 0, 1, 1]
        custom = CustomNation(nation_data)

        assert hasattr(custom, 'index')
        assert hasattr(custom, 'name')
        assert hasattr(custom, 'tagline')
        assert hasattr(custom, 'terrain')
        assert hasattr(custom, 'home_plane')
        assert hasattr(custom, 'team')
        assert hasattr(custom, 'iid')

    def test_custom_nation_terrain_profile_lookup(self):
        """CustomNation should look up terrain_profile from TERRAIN_PREFERENCES"""
        nation_data = [600, 'TestNation', 'Test Tagline', 16, 0, 0, 1, 1]
        custom = CustomNation(nation_data)

        assert hasattr(custom, 'terrain_profile')
        assert isinstance(custom.terrain_profile, list)

    def test_custom_nation_layout_lookup(self):
        """CustomNation should look up layout from LAYOUT_PREFERENCES"""
        nation_data = [600, 'TestNation', 'Test Tagline', 16, 0, 0, 1, 1]
        custom = CustomNation(nation_data)

        assert hasattr(custom, 'layout')
        assert isinstance(custom.layout, list)

    def test_custom_nation_home_plane(self):
        """CustomNation home_plane should be set from data"""
        nation_data = [600, 'TestNation', 'Test Tagline', 16, 0, 0, 2, 1]  # Plane 2 (underground)
        custom = CustomNation(nation_data)

        assert custom.home_plane == 2

    def test_custom_nation_iid_initialized_none(self):
        """CustomNation.iid should start as None"""
        nation_data = [600, 'TestNation', 'Test Tagline', 16, 0, 0, 1, 1]
        custom = CustomNation(nation_data)

        assert custom.iid is None


class TestGenericNationCreation:
    """Test GenericNation class for generic nation creation"""

    def test_generic_nation_creation(self):
        """GenericNation should create from generic nation data"""
        # Format: [terrain, terrain_index, layout_index, home_plane, team]
        nation_data = [16, 0, 0, 1, 1]
        generic = GenericNation(nation_data)

        assert generic is not None
        assert generic.terrain == 16
        assert generic.home_plane == 1

    def test_generic_nation_name(self):
        """GenericNation should have default name 'Generic Nation'"""
        nation_data = [16, 0, 0, 1, 1]
        generic = GenericNation(nation_data)

        assert generic.name == 'Generic Nation'

    def test_generic_nation_attributes(self):
        """GenericNation should have all expected attributes"""
        nation_data = [16, 0, 0, 1, 1]
        generic = GenericNation(nation_data)

        assert hasattr(generic, 'terrain')
        assert hasattr(generic, 'terrain_profile')
        assert hasattr(generic, 'layout')
        assert hasattr(generic, 'home_plane')
        assert hasattr(generic, 'team')
        assert hasattr(generic, 'name')
        assert hasattr(generic, 'iid')

    def test_generic_nation_terrain_profile_lookup(self):
        """GenericNation should look up terrain_profile from TERRAIN_PREFERENCES"""
        nation_data = [16, 0, 0, 1, 1]
        generic = GenericNation(nation_data)

        assert isinstance(generic.terrain_profile, list)

    def test_generic_nation_layout_lookup(self):
        """GenericNation should look up layout from LAYOUT_PREFERENCES"""
        nation_data = [16, 0, 0, 1, 1]
        generic = GenericNation(nation_data)

        assert isinstance(generic.layout, list)

    def test_generic_nations_all_have_default_name(self):
        """All GenericNation instances should share the default name"""
        generic1 = GenericNation([16, 0, 0, 1, 1])
        generic2 = GenericNation([32, 1, 1, 2, 2])

        assert generic1.name == generic2.name == 'Generic Nation'


class TestNationDataIntegrity:
    """Test that nation data attributes are valid and consistent"""

    def test_nation_home_plane_is_valid(self):
        """All nations should have valid home_plane (1 or 2)"""
        test_nations = [[11, 1], [31, 2], [15, 3]]

        for nation_data in test_nations:
            nation = Nation(nation_data)
            assert nation.home_plane in [1, 2], f"Invalid home_plane: {nation.home_plane}"

    def test_nation_terrain_profile_has_required_keys(self):
        """Nation terrain_profile should be a list with data"""
        nation = Nation([11, 1])

        assert isinstance(nation.terrain_profile, list)
        assert len(nation.terrain_profile) > 0

    def test_nation_layout_has_required_keys(self):
        """Nation layout should be a list with data"""
        nation = Nation([11, 1])

        assert isinstance(nation.layout, list)
        assert len(nation.layout) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
