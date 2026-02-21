import numpy as np
import noise as ns

def make_noise_array(map_size, scale):
    x_size, y_size = map_size
    noise_array = np.zeros((x_size, y_size, 2), dtype=np.float32)
    base_x, base_y = np.random.randint(1, 10000, size=2, dtype=np.int32)
    for x in range(x_size):
        for y in range(y_size):
            vx, vy = x/x_size, y/y_size
            noise_array[x, y, 0] = ns.snoise2(vx, vy, octaves=7, persistence=0.90, lacunarity=2.0, repeatx=1, repeaty=1, base=base_x)
            noise_array[x, y, 1] = ns.snoise2(vx, vy, octaves=7, persistence=0.90, lacunarity=2.0, repeatx=1, repeaty=1, base=base_y)
    return noise_array * scale
