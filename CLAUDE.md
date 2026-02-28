# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PointNet++ Workshop** is an educational Jupyter notebook-based workshop teaching point cloud processing and deep learning for survey engineering students. The project includes an interactive notebook, a web-based presentation, and a trained PyTorch model.

### Key Components
- **Main notebook**: `pointnet2_workshop.ipynb` - 9-section workshop with hands-on code
- **Web presentation**: `index.html` - Retro 16-bit styled slide deck
- **Trained model**: `pointnet2_modelnet10.pth` - Pre-trained PointNet++ classifier
- **Assets**: `assets/` - Presentation images (diagrams, screenshots, visualizations)

## Development Environment Setup

### Dependencies
The project uses **uv** for Python package management. All dependencies are specified in `pyproject.toml`:

```toml
requires-python = ">=3.12"
key-dependencies:
  - torch==2.8.0
  - torch-geometric>=2.7.0
  - torch-scatter>=2.1.2
  - plotly>=6.5.2
  - matplotlib>=3.10.8
  - scikit-learn>=1.8.0
  - jupyter/ipykernel (for notebook execution)
```

### Quick Start
```bash
# Install dependencies using uv (preferred)
uv sync

# Or use pip directly
pip install -e .

# Activate venv if using standard pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r <requirements>  # If uv lock is converted
```

### Running the Notebook
```bash
# Start Jupyter server
jupyter notebook

# Or use VS Code with Jupyter extension
# Open pointnet2_workshop.ipynb directly
```

**Google Colab** (recommended for students):
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `pointnet2_workshop.ipynb`
3. Run first cell to install PyTorch + dependencies
4. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU

## Project Structure

```
.
├── pointnet2_workshop.ipynb      # Main 9-section educational notebook
├── index.html                    # Web-based presentation (16-bit aesthetic)
├── pointnet2_modelnet10.pth      # Pre-trained PointNet++ model (~1.2M params)
├── pyproject.toml                # Python project configuration (uv-managed)
├── uv.lock                       # Dependency lock file
├── README.md                     # User-facing documentation
├── IMAGE_SUGGESTIONS.md          # Guide for enhancing presentation visuals
├── assets/                       # Presentation images
│   ├── pointnet.png             # PointNet architecture diagram
│   ├── pointnet++.png           # PointNet++ architecture diagram
│   ├── slide9_img2.png          # CloudCompare screenshot
│   ├── file format.png          # Point cloud file formats
│   └── ...                      # Other presentation assets
├── data/                        # (Auto-generated during notebook execution)
│   └── ModelNet10/              # Downloaded dataset cache
└── .venv/                       # (Optional) Python virtual environment
```

## Notebook Structure & Architecture

The notebook (`pointnet2_workshop.ipynb`) is divided into 9 progressive sections:

### Section 1-3: Foundations (40 minutes)
- **Setup & GPU configuration**: CUDA availability checks
- **Point cloud basics**: What are point clouds, why they're special (unordered, sparse, hierarchical)
- **Dataset exploration**: Load ModelNet10 (~1K shapes, 10 CAD object categories)
- **Preprocessing pipeline**: Normalization, sampling to fixed 1024 points

### Section 4: Architecture (25 minutes)
- **PointNet++ explained**: Visual diagrams of hierarchical abstraction
- **Set Abstraction (SA) modules**: FPS (Farthest Point Sampling), local feature grouping
- **Model walkthrough**: 1024 → 512 → 128 → 1 point hierarchy
- **Why it works**: Permutation invariance + hierarchical learning

### Section 5-7: Training & Analysis (75 minutes)
- **DataLoader setup**: PyTorch Geometric data structures
- **Training loop**: 30 epochs, accuracy tracking, loss visualization
- **Feature extraction**: PCA visualization of local features, t-SNE of global features
- **Results analysis**: Confusion matrix, per-class accuracy, confidence distributions

### Section 8-9: Interpretation & Extensions (20 minutes)
- **Interactive widget**: Select test samples, see real-time predictions
- **Applications**: Survey engineering use cases (BIM classification, terrain analysis, construction QA)
- **Extension ideas**: Segmentation, transfer learning, multi-task learning

### Key Technical Details
- **Model architecture**: 3-layer PointNet++ with SA blocks
- **Input**: 1024 3D points (x, y, z coordinates)
- **Output**: 10-class probabilities (ModelNet10 categories)
- **Training**: ~15-20 minutes on GPU, ~70-80% accuracy
- **Inference**: 50-100 samples/sec on GPU
- **Framework**: PyTorch + PyTorch Geometric

## Important Development Notes

### When Modifying the Notebook
1. **Section order matters**: Later sections depend on earlier cell definitions (model, dataset, etc.)
2. **GPU memory**: Default batch size is 32; reduce to 16 if OOM errors occur
3. **Dataset caching**: ModelNet10 downloads on first run (~300MB), then cached in `data/`
4. **Model checkpoint**: `pointnet2_modelnet10.pth` is loaded in training/inference sections

### Dependencies to Know
- **PyTorch Geometric**: Provides `Data`, `DataLoader`, and point cloud ops (NOT standard PyTorch)
- **torch-scatter/torch-sparse/torch-cluster**: PyG graph operations (require compilation)
- **Plotly**: Interactive 3D visualizations work in Jupyter, Colab, and browsers
- **Matplotlib/Seaborn**: Fallback for static plots and confusion matrices

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ImportError: No module named torch_geometric` | Run: `pip install torch-geometric torch-scatter torch-sparse torch-cluster` with correct PyTorch version |
| CUDA not available in Colab | Runtime → Change runtime type → GPU hardware accelerator |
| Out of memory errors | Reduce batch size (32 → 16), or use smaller dataset subset |
| Slow first run | ModelNet10 dataset downloads on first execution; subsequent runs use cache |
| Plotly visualization fails | Browser JavaScript must be enabled; Matplotlib fallback works |

## Presentation (index.html)

The web-based presentation uses:
- **Retro 16-bit pixel aesthetic** with custom CSS
- **Scan line effects** for CRT feel
- **Dark/light mode toggle** in header
- **Smooth scroll snap** between slides
- **Embedded images**: Architecture diagrams, CloudCompare screenshots, reference papers

To modify slides:
1. Edit HTML directly (uses inline `<style>` and SVG/img elements)
2. Add/replace images in `assets/` folder and update `src` paths
3. See `IMAGE_SUGGESTIONS.md` for enhancement ideas

## Key Commands & Workflows

### For Instructors/Maintainers
```bash
# View notebook (requires Jupyter)
jupyter notebook pointnet2_workshop.ipynb

# Run all cells (headless execution)
jupyter nbconvert --to notebook --execute pointnet2_workshop.ipynb

# Check presentation in browser
open index.html  # macOS
xdg-open index.html  # Linux
start index.html  # Windows
```

### For Students (Colab)
1. Upload notebook + dependencies to Colab
2. First cell installs PyTorch + PyG + visualization libs
3. Run sections sequentially (dependencies build up)

### Updating Dependencies
```bash
# Using uv
uv add <package-name>
uv lock  # Update uv.lock

# Or edit pyproject.toml directly and run:
uv sync
```

## Survey Engineering Context

This workshop connects point cloud deep learning to real-world applications:

- **Building component classification**: Automatically label BIM point clouds
- **Terrain analysis**: Separate vegetation, ground, buildings from LiDAR
- **Urban furniture detection**: Identify poles, benches, barriers in mobile mapping
- **Construction monitoring**: Compare design vs. as-built point clouds
- **Quality control**: Detect measurement errors and outliers

The PointNet++ architecture's hierarchical abstraction mirrors **Level of Detail (LoD)** concepts in GIS.

## Resources

- **Papers**: PointNet (2017), PointNet++ (2018) on arXiv
- **Official docs**: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- **Datasets**: ModelNet10/40, ShapeNet, ScanNet, S3DIS
- **Related**: Point cloud segmentation, instance segmentation, 3D object detection

## Future Extensions

The notebook can be extended with:
- **Semantic segmentation**: Label each point (not the whole cloud)
- **Custom datasets**: Import your own survey/LiDAR data
- **Transfer learning**: Fine-tune on domain-specific data
- **Instance segmentation**: Identify individual objects in scenes
- **Larger models**: PointNet++ with more SA blocks, deeper MLPs
