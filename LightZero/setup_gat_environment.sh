#!/bin/bash
# 
# GAT Environment Setup Script for Apptainer/Singularity
# Usage: ./setup_gat_environment.sh [container_path]
#

set -e

CONTAINER_PATH=${1:-""}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTZERO_ROOT="$SCRIPT_DIR"

echo "Setting up GAT environment for LightZero..."
echo "LightZero root: $LIGHTZERO_ROOT"

# Function to run commands in container or locally
run_cmd() {
    if [ -n "$CONTAINER_PATH" ]; then
        echo "Running in container: $1"
        apptainer exec --nv "$CONTAINER_PATH" bash -c "cd $LIGHTZERO_ROOT && $1"
    else
        echo "Running locally: $1"
        bash -c "cd $LIGHTZERO_ROOT && $1"
    fi
}

# Check if we're in a container environment
if [ -n "$CONTAINER_PATH" ]; then
    echo "Using Apptainer container: $CONTAINER_PATH"
    
    # Verify container exists
    if [ ! -f "$CONTAINER_PATH" ]; then
        echo "Error: Container file not found: $CONTAINER_PATH"
        exit 1
    fi
else
    echo "Running in local environment"
fi

# Install LightZero in development mode
echo "Installing LightZero..."
run_cmd "pip install -e ."

# Install GAT dependencies
echo "Installing GAT dependencies..."
run_cmd "pip install -r requirements-gat.txt"

# Test installation
echo "Testing GAT dependencies..."
run_cmd "python test_gat_dependencies.py"

# Verify GAT models can be imported
echo "Verifying GAT model imports..."
run_cmd "python -c \"
import sys
sys.path.insert(0, '.')
try:
    from lzero.model.gat_stochastic_muzero_model import GATStochasticMuZeroModel
    print('✓ GATStochasticMuZeroModel import successful')
except Exception as e:
    print(f'✗ GATStochasticMuZeroModel import failed: {e}')
    
try:
    from lzero.model.gat_muzero_model import GATMuZeroModel  
    print('✓ GATMuZeroModel import successful')
except Exception as e:
    print(f'✗ GATMuZeroModel import failed: {e}')
\""

echo "GAT environment setup completed!"
echo ""
echo "Usage examples:"
if [ -n "$CONTAINER_PATH" ]; then
    echo "  # Run GAT training in container:"
    echo "  apptainer exec --nv $CONTAINER_PATH python $LIGHTZERO_ROOT/train_gat_stochastic_3x3.py"
    echo ""
    echo "  # Run with custom config:"
    echo "  apptainer exec --nv $CONTAINER_PATH python $LIGHTZERO_ROOT/zoo/game_2048/config/gat_stochastic_2048_config.py"
else
    echo "  # Run GAT training:"
    echo "  python train_gat_stochastic_3x3.py"
    echo ""
    echo "  # Run with custom config:"
    echo "  python zoo/game_2048/config/gat_stochastic_2048_config.py"
fi
