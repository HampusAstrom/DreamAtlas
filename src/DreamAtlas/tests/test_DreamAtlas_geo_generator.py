import numpy as np
from DreamAtlas.generators.DreamAtlas_geo_generator import cleanup_isolated_pixels

def test_cleanup_isolated_pixels_basic():
    # Create a small pixel map with an isolated pixel
    pixel_map = np.array([
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1]
    ], dtype=np.int32)
    # The center pixel (2) is isolated
    cleaned = cleanup_isolated_pixels(pixel_map)
    # After cleanup, the center pixel should be replaced by a neighbor (1)
    assert cleaned[1, 1] == 1
    # All other pixels should remain unchanged
    for x in range(3):
        for y in range(3):
            if (x, y) != (1, 1):
                assert cleaned[x, y] == 1

if __name__ == '__main__':
    test_cleanup_isolated_pixels_basic()
    print('cleanup_isolated_pixels test passed!')
