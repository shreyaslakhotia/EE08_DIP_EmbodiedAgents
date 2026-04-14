#!/bin/bash

# =================================================================
# STANKY RUNNER SCRIPT
# =================================================================
# Usage: ./start_stanky.sh [/path/to/venv] [/path/to/project]
# =================================================================

# 1. CONFIGURATION (Edit these defaults or pass as arguments)
# -----------------------------------------------------------------
# Default Virtual Environment Path
VENV_PATH="${1:-./venv}" 

# Default Project Directory (where stanky.py lives)
PROJECT_DIR="${2:-$(pwd)}"
# -----------------------------------------------------------------

# Resolve absolute paths
ABS_VENV_PATH=$(realpath "$VENV_PATH")
ABS_PROJECT_DIR=$(realpath "$PROJECT_DIR")

echo "---------------------------------------"
echo "🚀 Starting MotivAI (Stanky)..."
echo "📂 Project Dir: $ABS_PROJECT_DIR"
echo "🐍 Venv Path:    $ABS_VENV_PATH"
echo "---------------------------------------"

# 2. Navigate to the project directory
if cd "$ABS_PROJECT_DIR"; then
    # 3. Check for the activation script
    if [ -f "$ABS_VENV_PATH/bin/activate" ]; then
        # 4. Activate and Run
        source "$ABS_VENV_PATH/bin/activate"
        
        # Check if stanky.py exists in the folder
        if [ -f "stanky.py" ]; then
            python3 stanky.py
        else
            echo "❌ Error: 'stanky.py' not found in $ABS_PROJECT_DIR"
        fi
        
        # 5. Cleanup
        deactivate
    else
        echo "❌ Error: Virtual environment not found at $ABS_VENV_PATH/bin/activate"
        echo "Usage: $0 [/path/to/venv] [/path/to/project_dir]"
    fi
else
    echo "❌ Error: Could not enter directory $ABS_PROJECT_DIR"
fi
