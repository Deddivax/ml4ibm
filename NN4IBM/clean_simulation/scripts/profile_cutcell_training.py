from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_simulation.internal_nn_runtime.cutcell_ml.model import CutCellNet
from clean_simulation.internal_nn_runtime.cutcell_ml.training import composite_loss


CHECKPOINT = ROOT / "clean_simulation" / "outputs" / "checkpoints" / (
    "source_sdf_rebuilt_queso_d10_f1024_all_crystals_lr5e6_calibration.pt"
)


def measure_batch(batch_size: int) -> None:
    model = CutCellNet(
        fusion_dim=1024,
        n_residual_blocks=10,
        trunk_type="resnet",
        residual_scaling="rezero",
        spd_mode="cholesky",
        use_reduced_null_space=True,
        matrix_size=24,
        n_distance_nodes=8,
    ).cuda()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cuda", weights_only=False)["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-6)
    d = torch.randn(batch_size, 8, device="cuda")
    nu = torch.full((batch_size,), 0.3, device="cuda")
    k_ref = torch.randn(batch_size, 24, 24, device="cuda")
    cut_class = torch.zeros(batch_size, dtype=torch.long, device="cuda")

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = composite_loss(model, d, nu, k_ref, cut_class=cut_class)[0]
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = composite_loss(model, d, nu, k_ref, cut_class=cut_class)[0]
        loss.backward()
        optimizer.step()
    end.record()
    end.synchronize()
    print(f"batch={batch_size}: {start.elapsed_time(end) / 5:.1f} ms/step", flush=True)

    del optimizer, model, d, nu, k_ref, cut_class, loss
    torch.cuda.empty_cache()


def profile_batch(batch_size: int) -> None:
    model = CutCellNet(fusion_dim=1024, n_residual_blocks=10, trunk_type="resnet", residual_scaling="rezero", spd_mode="cholesky", use_reduced_null_space=True, matrix_size=24, n_distance_nodes=8).cuda()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cuda", weights_only=False)["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-6)
    d = torch.randn(batch_size, 8, device="cuda")
    nu = torch.full((batch_size,), 0.3, device="cuda")
    k_ref = torch.randn(batch_size, 24, 24, device="cuda")
    cut_class = torch.zeros(batch_size, dtype=torch.long, device="cuda")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = composite_loss(model, d, nu, k_ref, cut_class=cut_class)[0]
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=25), flush=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for batch_size in (512, 768, 1024):
        try:
            measure_batch(batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"batch={batch_size}: OOM", flush=True)
    profile_batch(512)