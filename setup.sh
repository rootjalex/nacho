#!/usr/bin/env bash
#
# setup.sh — Set up nacho compiler and runtime on a new machine.
#
# Usage:
#   ./setup.sh              # Full setup (compiler + runtime)
#   ./setup.sh --compiler   # Compiler only (no CUDA needed)
#   ./setup.sh --runtime    # Runtime only (needs CUDA)
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---all}"
ENV_NAME="nacho"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
error() { echo -e "${RED}[setup]${NC} $*" >&2; }
die()   { error "$@"; exit 1; }

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

ensure_conda() {
    if ! command -v conda &>/dev/null; then
        die "conda not found. Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
    fi
    info "Found $(conda --version)"
}

setup_env() {
    ensure_conda

    if conda env list | grep -qw "^${ENV_NAME} "; then
        warn "Conda env '$ENV_NAME' already exists."
        echo -e "  [u] Update — install into existing env"
        echo -e "  [r] Recreate — delete and start fresh"
        echo -e "  [a] Abort"
        read -rp "Choice [u/r/a]: " choice
        case "$choice" in
            r|R)
                info "Removing existing env '$ENV_NAME'..."
                conda env remove -y -n "$ENV_NAME"
                info "Creating conda env '$ENV_NAME' (Python 3.12)..."
                conda create -y -n "$ENV_NAME" python=3.12
                ;;
            u|U)
                info "Updating existing env '$ENV_NAME'..."
                ;;
            *)
                die "Aborted."
                ;;
        esac
    else
        info "Creating conda env '$ENV_NAME' (Python 3.12)..."
        conda create -y -n "$ENV_NAME" python=3.12
    fi

    # Activate in this script's context
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    info "Active env: $CONDA_DEFAULT_ENV ($(python --version))"
}

load_cuda() {
    if command -v nvcc &>/dev/null; then
        info "Found $(nvcc --version | tail -1)"
        return
    fi

    # Source the module system if not already available (needed in non-interactive shells)
    if ! command -v module &>/dev/null; then
        for init in /etc/profile.d/modules.sh /usr/share/modules/init/bash; do
            if [[ -f "$init" ]]; then
                source "$init"
                break
            fi
        done
    fi

    # Try module load if available (common on clusters)
    if command -v module &>/dev/null; then
        local cuda_mod
        cuda_mod=$(module avail cuda 2>&1 | grep -oP 'cuda/\S+' | sort -V | tail -1 || true)
        if [[ -n "$cuda_mod" ]]; then
            info "Loading module $cuda_mod"
            module load "$cuda_mod"
            if command -v nvcc &>/dev/null; then
                info "Found $(nvcc --version | tail -1)"
                return
            fi
        fi
    fi

    die "nvcc not found. Install CUDA toolkit or 'module load cuda'."
}

# ---------------------------------------------------------------------------
# Compiler setup
# ---------------------------------------------------------------------------

setup_compiler() {
    info "=== Setting up nacho compiler ==="

    # Compiler needs CMake 3.30+ — install via conda if system version is too old
    local need_cmake=false
    if command -v cmake &>/dev/null; then
        local cmake_ver
        cmake_ver=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
        if awk "BEGIN{exit(!($cmake_ver < 3.30))}"; then
            warn "System CMake is $cmake_ver, need 3.30+. Installing via conda..."
            need_cmake=true
        fi
    else
        need_cmake=true
    fi

    if $need_cmake; then
        conda install -y -c conda-forge cmake
        info "Installed cmake $(cmake --version | head -1 | awk '{print $3}') via conda"
    else
        info "Found cmake $(cmake --version | head -1 | awk '{print $3}')"
    fi

    info "Configuring nacho compiler..."
    cmake -S "$REPO_DIR" -B "$REPO_DIR/build"

    info "Building nacho compiler..."
    cmake --build "$REPO_DIR/build" -j"$(nproc)"

    info "Compiler built: $REPO_DIR/build/compiler"
}

# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------

setup_runtime() {
    info "=== Setting up runtime ==="

    load_cuda

    if command -v nvidia-smi &>/dev/null; then
        local gpu
        gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        info "GPU: ${gpu:-unknown}"
    fi

    # Ensure CMake is new enough (cuco requires 3.23.1+, scikit-build-core uses system cmake)
    local need_cmake=false
    if command -v cmake &>/dev/null; then
        local cmake_ver
        cmake_ver=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
        if awk "BEGIN{exit(!($cmake_ver < 3.24))}"; then
            warn "System CMake is $cmake_ver, runtime deps need 3.23.1+. Installing via conda..."
            need_cmake=true
        fi
    else
        need_cmake=true
    fi

    if $need_cmake; then
        conda install -y -c conda-forge cmake
        info "Installed cmake $(cmake --version | head -1 | awk '{print $3}') via conda"
    fi

    info "Installing Python dependencies..."
    pip install --quiet numpy pandas matplotlib tqdm

    # Install PyTorch with CUDA support
    info "Installing PyTorch (CUDA)..."
    pip install --quiet torch

    info "Installing runtime package (compiles CUDA — may take a few minutes)..."
    pip install --verbose "$REPO_DIR/runtime"

    info "Verifying runtime import..."
    if python -c "import nanobind_cuda_example; print('  nanobind_cuda_example OK')"; then
        info "Runtime installed successfully."
    else
        error "Failed to import nanobind_cuda_example. Check build output above."
        return 1
    fi

    # Check for SuiteSparse matrices
    local ss_dir="/scratch/atharva/suitesparse"
    if [[ -d "$ss_dir" ]]; then
        info "SuiteSparse matrices found at $ss_dir"
    else
        warn "SuiteSparse matrices not found at $ss_dir"
        warn "Benchmarks using parser.py need matrices at that path."
        warn "Symlink or update the path in runtime/tests/parser.py."
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "$MODE" in
    --compiler)
        setup_env
        setup_compiler
        ;;
    --runtime)
        setup_env
        setup_runtime
        ;;
    --all|"")
        setup_env
        setup_compiler || warn "Compiler setup failed (see above), continuing with runtime..."
        setup_runtime
        ;;
    -h|--help)
        cat <<'USAGE'
Usage: ./setup.sh [--compiler|--runtime|--all]

  --compiler   Build the nacho compiler only (no CUDA needed)
  --runtime    Install the CUDA runtime + Python package (needs CUDA + GPU)
  --all        Both (default)

Creates a conda environment called 'nacho' with all dependencies.
After setup, activate it with: conda activate nacho
USAGE
        exit 0
        ;;
    *)
        die "Unknown option: $MODE. Use --help for usage."
        ;;
esac

echo ""
info "Done. To use this environment in the future:"
info "  conda activate $ENV_NAME"
