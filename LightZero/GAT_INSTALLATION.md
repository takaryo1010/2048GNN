# GAT Installation Guide for Apptainer/Singularity Environments

This guide explains how to set up LightZero with GAT (Graph Attention Network) support in apptainer/singularity containers.

## Prerequisites

- PyTorch installed with CUDA support (if using GPU)
- Python >= 3.7

## Installation Steps

### 1. Basic LightZero Installation

```bash
# Clone and install LightZero
git clone https://github.com/opendilab/LightZero.git
cd LightZero
pip install -e .
```

### 2. Install GAT Dependencies

GAT models require PyTorch Geometric and related packages:

```bash
# Install PyTorch Geometric dependencies
pip install torch-geometric>=2.3.0
pip install torch-scatter>=2.1.0 
pip install torch-sparse>=0.6.17
pip install torch-cluster>=1.6.0

# Or install all at once
pip install -r requirements-gat.txt
```

### 3. Verify Installation

```bash
# Test if GAT dependencies are properly installed
python test_gat_dependencies.py
```

### 4. Container-Specific Considerations

#### For Apptainer/Singularity:

1. **CUDA Libraries**: Ensure CUDA libraries are available in the container or bind them from the host:
   ```bash
   apptainer exec --nv your_container.sif python your_gat_script.py
   ```

2. **Host Path Binding**: If importing GAT models from host-side paths (as in gat_stochastic_2048_config.py):
   ```bash
   apptainer exec --bind /host/path:/container/path your_container.sif python config.py
   ```

3. **Environment Variables**: Set proper PYTHONPATH if needed:
   ```bash
   export PYTHONPATH="/container/LightZero:$PYTHONPATH"
   ```

### 5. Common Issues and Solutions

#### Issue: "ModuleNotFoundError: No module named 'torch_geometric'"
Solution: Install PyTorch Geometric dependencies as shown in step 2.

#### Issue: CUDA compatibility errors
Solution: Ensure PyTorch version matches CUDA version and install corresponding torch-geometric versions.

#### Issue: Import errors from host-side paths
Solution: Check that host paths are properly bound and accessible in container.

### 6. Container Definition Example

For creating your own container with GAT support:

```bash
# Install base requirements
pip install -r requirements.txt

# Install GAT requirements  
pip install -r requirements-gat.txt

# Verify installation
python test_gat_dependencies.py
```

### 7. Running GAT Models

```bash
# Example: Run GAT-based StochasticMuZero on 3x3 board
python train_gat_stochastic_3x3.py

# Example: Run with custom configuration
python -c "
import sys
sys.path.append('/path/to/LightZero')
from zoo.game_2048.config.gat_stochastic_2048_config import *
"
```

## Performance Notes

- GAT models require more memory than standard CNN models
- GPU acceleration is recommended for larger graphs
- Consider batch size adjustments based on available memory
