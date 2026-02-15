"""
Pytest configuration file
Ensures DreamAtlas can be imported from tests
"""
import sys
from pathlib import Path

# Add the parent directory (procedural_generation) to the Python path
# so that 'import DreamAtlas' can find the DreamAtlas package
dreamatlas_parent = Path(__file__).parent.parent
sys.path.insert(0, str(dreamatlas_parent))
