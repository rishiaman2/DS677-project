# Sourced by every Wulver job script, and by you on the login node before preparing data.
# Follows the course guide "Getting Started on Wulver": all work lives in the course directory,
# and jobs use the ds677 conda environment built there with the guide's install/verify jobs
# (it already has torch, torchvision, transformers, datasets, pyyaml, scikit-learn, matplotlib).
COURSE_DIR=/course/2026/fall/ds/677/ikoutis/rv44
export REPO_DIR=${REPO_DIR:-$COURSE_DIR/DS677-project}
export ENV_PREFIX=${ENV_PREFIX:-$COURSE_DIR/envs/ds677}
export HF_HOME=$COURSE_DIR/hf_cache
module load CUDA
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$REPO_DIR"
