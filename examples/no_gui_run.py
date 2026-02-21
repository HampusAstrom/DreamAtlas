import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path so DreamAtlas can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from DreamAtlas import *

# Default config file (can be overridden by command line argument)
default_config = Path(__file__).parent.parent / "databases" / "12_player_ea_test.dream"

def main(config_path=default_config):
    settings = DreamAtlasSettings(index=0)
    settings.load_file(filename=str(config_path))
    map_obj = generator_dreamatlas(settings=settings)
    map_obj.plot()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DreamAtlas map generation and plot without GUI.")
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to config .dream file")
    args = parser.parse_args()
    main(args.config)
