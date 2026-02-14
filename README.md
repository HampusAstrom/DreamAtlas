# DreamAtlas
Dominions 6 map generator

DreamAtlas is an open-source Python library for Dominions 6 map editing and generation.  DreamAtlas allows users to seamlessly interact with Dominions maps through Python code. Equipped with groundbreaking map generation algorithms and tools for artists that will allow maps to be made better than ever before.

I've always loved fantasy, and a major part of fantasy for me is elaborate maps, epic mountain ranges, rivers and so on. I also love competition and multiplayer strategy games. Dominions has never really brought this together for me, most maps are either focused on nice geography or good balance (or neither in vanilla), so I decided to make my own map generator to make the kind of maps I'd like to play on. This meant going to the drawing board and really asking what makes a good dominions map and what maths you need to accomplish that? Further I wanted to make a tool not just to make maps, but to make map generators and support people making maps from custom art, as a result the DreamAtlas generator is essentially just a script calling functions and classes from the DreamAtlas library, anyone with some python knowledge could make their own mapmaker with DreamAtlas using the generalised tools, classes and methods it comes with.

## Installation

### Verified installation method: Conda (Anaconda/Miniconda)

I installed this repo using conda, so that is the only currently verified installation method.This sets up an isolated Python environment for DreamAtlas, ensuring all dependencies are properly installed without conflicts with your system Python.

#### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) installed

#### Steps

OBS: replace / with \\ as needed for unix/windows style terminals

0. **(optional) Extra wrapping directory**
   For some (import) reason, imports only work propperly when called from outside the git folder, like "python DreamAtlas/examples/run.py". So consider wrapping the git repo in a parent folder like: "\<parent\>/DreamAtlas" before proceeding.

1. **Create a new conda environment:**
   ```bash
   conda create -n dreamatlas python=3.11
   ```
   (Replace `3.11` with your preferred Python version if needed)

2. **Activate the environment:**
   ```bash
   conda activate dreamatlas
   ```

3. **Navigate to the DreamAtlas directory:**
   ```bash
   cd path/to/DreamAtlas
   ```

4. **Install DreamAtlas in development mode:**
   ```bash
   pip install -e .
   ```
   This installs the package with all dependencies listed in `setup.py` in editable mode, allowing you to modify the code and have changes reflected immediately.

5. **Verify installation:**
   ```bash
   python -c "import DreamAtlas; print('DreamAtlas installed successfully!')"
   ```

6. **Run the GUI application:**
   ```bash
   # From the parent directory of DreamAtlas
   python DreamAtlas/examples/run.py
   ```

### Alternative: Virtual Environment (venv)

If you prefer not to use conda, Python's built-in `venv` module works as well:

```bash
# Create virtual environment
python -m venv dreamatlas_env

# Activate (Linux/Mac)
source dreamatlas_env/bin/activate

# Activate (Windows)
dreamatlas_env\Scripts\activate

# Install DreamAtlas
pip install -e .
```

### Alternative: Direct Installation (Not Recommended)

For a system-wide installation without isolation:

```bash
pip install -e .
```

**Note:** This may cause conflicts with other packages. Using conda or venv is strongly recommended.

## Dependencies

DreamAtlas requires the following packages (automatically installed via `setup.py`):
- numpy - Numerical computing
- scipy - Scientific computing
- matplotlib - Visualization
- networkx - Graph/network operations
- minorminer - Graph embedding
- ttkbootstrap - Modern themed GUI toolkit
- noise - Perlin noise generation
- numba - JIT compilation for performance
- Pillow - Image processing
