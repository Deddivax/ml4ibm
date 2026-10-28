"""Portable remote-training control helpers for notebooks and operations."""

from __future__ import annotations

import base64
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s*/\s*\d+\s+\| Train: ([0-9.eE+-]+) \| "
    r"Val: ([0-9.eE+-]+).*?Lr: ([0-9.eE+-]+)"
)


@dataclass(frozen=True, slots=True)
class RemoteTrainingSettings:
    """Environment-backed connection and artifact settings for a training host."""

    host: str
    ssh_key: Path
    repository: str
    run_root: str
    python: str
    torchrun: str
    training_entrypoint: str
    data_path: str

    @classmethod
    def from_environment(cls) -> "RemoteTrainingSettings":
        return cls(
            host=os.environ.get("CUT_CELL_REMOTE_HOST", ""),
            ssh_key=Path(
                os.environ.get("CUT_CELL_SSH_KEY", Path.home() / ".ssh" / "cutcell-training.pem")
            ).expanduser(),
            repository=os.environ.get("CUT_CELL_REMOTE_REPO", "~/cutcell-training"),
            run_root=os.environ.get("CUT_CELL_RUN_ROOT", "clean_simulation/outputs/adaptive_training"),
            python=os.environ.get("CUT_CELL_REMOTE_PYTHON", "python3"),
            torchrun=os.environ.get("CUT_CELL_REMOTE_TORCHRUN", "torchrun"),
            training_entrypoint=os.environ.get(
                "CUT_CELL_TRAINING_ENTRYPOINT",
                "-m clean_simulation.scripts.training.train_model",
            ),
            data_path=os.environ.get(
                "CUT_CELL_DATA_PATH",
                "clean_simulation/outputs/data/exact_original_sdf_lattice_rebuilt_queso_i2_p1_all72.h5",
            ),
        )

    def remote_file(self, run_name: str, filename: str) -> str:
        return f"{self.repository}/{self.run_root}/{run_name}/{filename}"

    def run_shell(self, script: str, *, check: bool = True, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        """Run a non-interactive Bash script on the configured host."""
        if not self.host:
            raise RuntimeError(
                "Remote control is not configured. Set CUT_CELL_REMOTE_HOST and CUT_CELL_SSH_KEY, "
                "or run python -m clean_simulation.scripts.training.train_model locally."
            )
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        try:
            completed = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "ConnectTimeout=15",
                    "-o",
                    "BatchMode=yes",
                    "-i",
                    str(self.ssh_key),
                    self.host,
                    f"echo {encoded} | base64 -d | bash",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"remote_shell timed out after {timeout}s; SSH session likely stuck.") from error
        if check and completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed


def parse_epoch_log(log_text: str) -> list[dict[str, float | int]]:
    """Extract completed epoch metrics from the portable trainer log format."""
    return [
        {
            "epoch": int(epoch),
            "train_loss": float(train),
            "val_loss": float(validation),
            "learning_rate": float(learning_rate),
        }
        for epoch, train, validation, learning_rate in EPOCH_PATTERN.findall(log_text)
    ]


def build_launch_script(
    settings: RemoteTrainingSettings,
    training: dict,
    *,
    checkpoint: str | None,
) -> str:
    """Build the guarded notebook's remote training command without executing it."""
    optimizer = training.get("optimizer", "adamw").lower()
    use_amp = bool(training.get("use_amp", True))
    weight_decay = float(training.get("weight_decay", 1.0e-5))
    sgd_momentum = float(training.get("sgd_momentum", 0.9))
    hard_example_drop_fraction = float(training.get("hard_example_drop_fraction", 0.0))
    compile_mode = training.get("compile_mode", "none").lower()
    distributed = bool(training.get("distributed", False))
    scheduler_type = training.get("scheduler_type", "cosine").lower()
    preserve_normalizers = bool(training.get("preserve_estimated_input_normalizers", False))
    lbfgs_reset_history = bool(training.get("lbfgs_reset_history_per_batch", True))
    launcher = f"{settings.torchrun} --standalone --nproc_per_node=2" if distributed else settings.python
    cuda_devices = training.get("cuda_devices", "0,1" if distributed else "0")
    run_name = training["run_name"]
    return f'''\
set -euo pipefail
cd {settings.repository}
RUN_DIR={settings.run_root}/{run_name}
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"
export CUT_CELL_MODEL_TYPE={training['model_type']}
export CUT_CELL_FUSION_DIM={int(training['fusion_dim'])}
export CUT_CELL_RESIDUAL_BLOCKS={int(training['residual_blocks'])}
export CUT_CELL_N_ATTENTION_HEADS={int(training.get('n_attention_heads', 0))}
export CUT_CELL_BATCH_SIZE={int(training['batch_size'])}
export CUT_CELL_EPOCHS={int(training['epochs'])}
export CUT_CELL_MAX_WALL_SECONDS={int(3600 * training['wall_hours'])}
export CUT_CELL_INITIAL_LR={float(training['initial_lr'])}
export CUT_CELL_SCHEDULER_TYPE={scheduler_type}
export CUT_CELL_COSINE_ETA_MIN_FACTOR={float(training.get('eta_min_factor', 1.0))}
export CUT_CELL_SPIKE_THRESHOLD={float(training['spike_threshold'])}
export CUT_CELL_SPIKE_LR_REDUCTION_FACTOR={float(training['spike_lr_reduction_factor'])}
export CUT_CELL_MAX_SPIKE_RETRIES={int(training['max_spike_retries'])}
export CUT_CELL_OPTIMIZER={optimizer}
export CUT_CELL_SGD_MOMENTUM={sgd_momentum}
export CUT_CELL_WEIGHT_DECAY={weight_decay}
export CUT_CELL_USE_AMP={int(use_amp)}
export CUT_CELL_HARD_EXAMPLE_DROP_FRACTION={hard_example_drop_fraction}
export CUT_CELL_EXCLUDED_GEOMETRY_PREFIXES={','.join(training.get('excluded_geometry_prefixes', ()))!r}
export CUT_CELL_LBFGS_MAX_ITER={int(training.get('lbfgs_max_iter', 4))}
export CUT_CELL_LBFGS_HISTORY_SIZE={int(training.get('lbfgs_history_size', 20))}
export CUT_CELL_LBFGS_RESET_HISTORY_PER_BATCH={int(lbfgs_reset_history)}
export CUT_CELL_SEED={int(training['seed'])}
export CUT_CELL_SPLIT_SEED={int(training['seed'])}
export CUT_CELL_SHUFFLE_SEED={int(training['seed'])}
export CUT_CELL_TRANSFORMER_ORDERED_SDF_BRANCH={int(bool(training['ordered_sdf_branch']))}
export CUT_CELL_REFINED_SUBCELLS={int(training.get('refined_subcells', 1))}
export CUT_CELL_COMPILE_MODE={compile_mode}
export CUT_CELL_DISTRIBUTED={int(distributed)}
export CUT_CELL_PRESERVE_ESTIMATED_INPUT_NORMALIZERS={int(preserve_normalizers)}
export CUT_CELL_DATA_PATH={training.get('data_path', settings.data_path)!r}
export CUT_CELL_INIT_CHECKPOINT={checkpoint!r}
export CUT_CELL_OUTPUT_CHECKPOINT="$RUN_DIR/stage_01_best.pt"
export CUT_CELL_HISTORY_PATH="$RUN_DIR/stage_01_history.json"
nohup env CUDA_VISIBLE_DEVICES={cuda_devices} {launcher} {settings.training_entrypoint} > "$RUN_DIR/training.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/pid"
echo "Launched {'two-rank DDP' if distributed else 'single-process'} training in $RUN_DIR with PID $(cat \"$RUN_DIR/pid\") on physical GPU(s) {cuda_devices}."
'''


def build_refined_data_generation_script(
    settings: RemoteTrainingSettings,
    *,
    source_h5: str,
    output_path: str,
    refined_subcells: int,
    queso_python_path: str,
) -> str:
    """Build a fail-closed command for one true-SDF regenerated corpus."""
    if refined_subcells < 1:
        raise ValueError("refined_subcells must be at least one.")
    return f'''\
set -euo pipefail
cd {shlex.quote(settings.repository)}
test ! -e {shlex.quote(output_path)}
{shlex.quote(settings.python)} -m clean_simulation.scripts.regenerate_material_parity_dataset_aws \\
  --queso-python-path {shlex.quote(queso_python_path)} \\
    --source-h5 {shlex.quote(source_h5)} \
  --sdf-subcells {refined_subcells} \\
  --output-h5 {shlex.quote(output_path)}
'''


def build_inventory_script(settings: RemoteTrainingSettings) -> str:
    """Build the remote checkpoint, process, and GPU inventory command."""
    return f'''\
set -euo pipefail
cd {settings.repository}
{settings.python} - <<'PY'
import json
from pathlib import Path
root = Path("{settings.run_root}")
rows = []
for directory in sorted(path for path in root.iterdir() if path.is_dir()):
    checkpoint = directory / "stage_01_best.pt"
    history = directory / "stage_01_history.json"
    pid_file = directory / "pid"
    payload = {{}}
    if history.is_file():
        try:
            payload = json.loads(history.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    rows.append({{
        "run": directory.name,
        "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
        "checkpoint MiB": round(checkpoint.stat().st_size / 2**20, 2) if checkpoint.is_file() else None,
        "history epochs": len(payload.get("epochs", [])),
        "history best": payload.get("best_val_loss"),
        "pid": pid_file.read_text().strip() if pid_file.is_file() else None,
    }})
print(json.dumps(rows))
PY
printf '\nACTIVE\n'
ps -eo pid=,stat=,etime=,args= | grep -E '[t]raining\\.train_model|[t]raining\\.run|[t]rain.py|[r]un_d20_f4096_aws.py|[t]orchrun' || true
printf '\nGPU\n'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
'''


def build_interrupt_script(settings: RemoteTrainingSettings, run_name: str) -> str:
    """Build a guarded remote SIGTERM command for one recorded training run."""
    return f'''\
set -euo pipefail
cd {settings.repository}
PID_FILE={settings.run_root}/{run_name}/pid
test -f "$PID_FILE"
PID=$(cat "$PID_FILE")
COMMAND=$(ps -p "$PID" -o args= || true)
case "$COMMAND" in *training.train_model*|*training.run*|*train.py*|*run_d20_f4096_aws.py*|*torchrun*) ;; *) echo "Refusing: PID $PID is not the training launcher" >&2; exit 2;; esac
kill -TERM "$PID"
echo "Sent SIGTERM to $PID for {run_name}."
'''


def build_active_runs_script(settings: RemoteTrainingSettings) -> str:
    """Build a remote query returning recorded training processes that are alive."""
    return f'''\
cd {settings.repository}
{settings.python} - <<'PY'
import json
from pathlib import Path
root = Path("{settings.run_root}")
rows = []
for directory in sorted(path for path in root.iterdir() if path.is_dir()):
    pid_file = directory / "pid"
    if not pid_file.is_file():
        continue
    pid = pid_file.read_text().strip()
    if pid and Path("/proc", pid).exists():
        rows.append({{"run": directory.name, "pid": pid}})
print(json.dumps(rows))
PY
'''


def build_run_status_script(settings: RemoteTrainingSettings, run_name: str) -> str:
    """Build a remote query returning the recorded PID and active state for one run."""
    return f'''\
set -euo pipefail
cd {settings.repository}
PID=$(cat {settings.run_root}/{run_name}/pid)
test -n "$PID"
if test -d /proc/$PID; then echo ACTIVE; else echo FINISHED; fi
echo $PID
'''


def synchronize_training_log(
    settings: RemoteTrainingSettings,
    run_name: str,
    destination: Path,
    *,
    atomically: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Copy one remote training log into a local cache path without shell interpolation."""
    if not settings.host:
        raise RuntimeError(
            "Remote control is not configured. Set CUT_CELL_REMOTE_HOST and CUT_CELL_SSH_KEY, "
            "or run python -m clean_simulation.scripts.training.train_model locally."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part") if atomically else destination
    completed = subprocess.run(
        [
            "scp",
            "-q",
            "-i",
            str(settings.ssh_key),
            f"{settings.host}:{settings.remote_file(run_name, 'training.log')}",
            str(temporary),
        ],
        capture_output=True,
        text=True,
    )
    if not completed.returncode and atomically:
        temporary.replace(destination)
    return completed