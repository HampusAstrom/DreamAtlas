import sys
from pathlib import Path

# Add parent directory to path so DreamAtlas can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from DreamAtlas import *
from DreamAtlas.GUI import run_interface
import cProfile

if __name__ == "__main__":  # This is what runs in the .exe loading up the main interface which contains all the DreamAtlas functionality
    # cProfile.run('run_interface()', sort='cumulative')
    run_interface()