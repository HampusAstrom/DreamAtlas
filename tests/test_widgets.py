

import pytest
import tkinter
from DreamAtlas.GUI.widgets import InputWidget, VanillaNationWidget
from DreamAtlas.classes import DominionsMap

@pytest.fixture(scope="module")
def tk_root():
    root = tkinter.Tk()
    yield root
    # Only destroy root after all tests in this module are done
    root.destroy()

def test_input_widget_instantiation(tk_root):
    # Test InputWidget can be instantiated with minimal config
    # Minimal valid ui_config with required keys
    ui_config = {
        'label_frames': [],
        'attributes': {},
        'buttons': []
    }
    try:
        widget = InputWidget(master=tk_root, ui_config=ui_config, target_type=None)
    except tkinter.TclError as e:
        if 'display' in str(e).lower() or 'screen' in str(e).lower():
            pytest.skip(f"InputWidget instantiation skipped (no display): {e}")
        raise
    except Exception as e:
        raise

def test_vanilla_nation_widget_instantiation(tk_root):
    # Test VanillaNationWidget can be instantiated
    try:
        widget = VanillaNationWidget(master=tk_root)
    except tkinter.TclError as e:
        if 'display' in str(e).lower() or 'screen' in str(e).lower():
            pytest.skip(f"VanillaNationWidget instantiation skipped (no display): {e}")
        raise
    except Exception as e:
        raise
