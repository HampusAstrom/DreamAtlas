from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'DreamAtlas and FlowAtlas'
LONG_DESCRIPTION = '''DreamAtlas: The premier map editing and generation tool
FlowAtlas: Experimental flow-based map generation for DreamAtlas'''

# Setting up both DreamAtlas and FlowAtlas packages
setup(
    name="DreamAtlas",
    version=VERSION,
    author="Tlaloc",
    author_email="<youremail@email.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/plain",
    package_dir={"": "src"},
    packages=find_packages(where="src"),  # Finds both DreamAtlas and FlowAtlas
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'networkx',
        'minorminer',
        'ttkbootstrap',
        'noise',
        'numba',
        'Pillow',
        'geopandas',
        'libpysal',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
        ],
    },
    keywords=['procedural-generation', 'map-generation', 'voronoi', 'waveform collapse'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    project_urls={
        'Bug Reports': 'https://github.com/HampusAstrom/DreamAtlas/issues',
        'Source': 'https://github.com/HampusAstrom/DreamAtlas',
    },
)
