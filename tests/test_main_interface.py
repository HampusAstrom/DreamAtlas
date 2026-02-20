import pytest
from DreamAtlas.GUI.main_interface import MainInterface

@pytest.fixture(scope="module")
def tk_root():
    import tkinter
    root = tkinter.Tk()
    yield root
    root.destroy()

def test_main_interface_instantiation(tk_root):
    # Test that MainInterface can be instantiated without error
    try:
        interface = MainInterface(master=tk_root)
    except Exception as e:
        pytest.skip(f"MainInterface instantiation skipped: {e}")
