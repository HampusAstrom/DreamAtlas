import ast
import os
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

def find_colormaps_in_file(filepath):
    colormaps = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=filepath)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == 'imshow':
                for kw in node.keywords:
                    if kw.arg == 'cmap':
                        if isinstance(kw.value, ast.Str):
                            colormaps.add(kw.value.s)
                        elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            colormaps.add(kw.value.value)
    return colormaps

def test_colormaps_used_in_code():
    # List files to scan for imshow(..., cmap=...)
    files = [
        os.path.join(os.path.dirname(__file__), '../classes/class_map.py'),
        os.path.join(os.path.dirname(__file__), '../classes/class_region.py'),
    ]
    found_colormaps = set()
    for file in files:
        found_colormaps.update(find_colormaps_in_file(file))
    # Test each colormap found in code
    data = np.arange(100).reshape(10, 10)
    for cmap in found_colormaps:
        fig, ax = plt.subplots()
        try:
            ax.imshow(data, cmap=cmap)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"Colormap '{cmap}' found in code failed with error: {e}")
