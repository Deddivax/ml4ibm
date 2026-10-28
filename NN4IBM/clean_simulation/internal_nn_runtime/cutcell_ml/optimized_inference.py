from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import urllib.request
import warnings

import torch
import torch.nn as nn

from .feature_contract import REFINED_SDF_FEATURE_ORDER

from clean_simulation.internal_nn_runtime.cutcell_ml.model import CutCellNet


@dataclass(slots=True)
class DownloadSpec:
    """Remote artifact description for config + checkpoint downloads."""

    config_url: str
    checkpoint_url: str
    output_dir: Path
    config_filename: str = "model_config.json"
    checkpoint_filename: str = "model_checkpoint.pt"
    config_sha256: str | None = None
    checkpoint_sha256: str | None = None


@dataclass(slots=True)
class InferenceOptions:
    """Performance options for runtime inference engine."""

    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    num_threads: int = 1
    num_interop_threads: int = 1
    use_jit: bool = True
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    warmup_iters: int = 32
    warmup_batch_size: int = 256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    url: str,
    target_path: Path,
    *,
    expected_sha256: str | None = None,
    timeout: float = 60.0,
    overwrite: bool = False,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not overwrite:
        if expected_sha256 is not None:
            got = _sha256_file(target_path)
            if got.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"Existing file hash mismatch for {target_path}: "
                    f"expected={expected_sha256}, got={got}"
                )
        return target_path

    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    target_path.write_bytes(data)

    if expected_sha256 is not None:
        got = _sha256_file(target_path)
        if got.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"Downloaded file hash mismatch for {target_path}: "
                f"expected={expected_sha256}, got={got}"
            )

    return target_path


def download_nn_configuration(
    config_url: str,
    output_path: Path,
    *,
    sha256: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a model configuration JSON file."""
    return _download_file(
        config_url,
        output_path,
        expected_sha256=sha256,
        overwrite=overwrite,
    )


def download_model_artifacts(spec: DownloadSpec, *, overwrite: bool = False) -> dict[str, Path]:
    """Download config and checkpoint artifacts into spec.output_dir."""
    out_dir = Path(spec.output_dir)
    config_path = _download_file(
        spec.config_url,
        out_dir / spec.config_filename,
        expected_sha256=spec.config_sha256,
        overwrite=overwrite,
    )
    checkpoint_path = _download_file(
        spec.checkpoint_url,
        out_dir / spec.checkpoint_filename,
        expected_sha256=spec.checkpoint_sha256,
        overwrite=overwrite,
    )
    return {
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
    }


class _BatchedWrapper(nn.Module):
    """Wrapper that enforces batched input shape for trace/JIT friendliness."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, d: torch.Tensor, nu: torch.Tensor | None = None) -> torch.Tensor:
        if d.ndim == 1:
            d = d.unsqueeze(0)
        return self.base(d, nu)


class FastCutCellInference:
    """High-performance inference runtime for CutCellNet checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        options: InferenceOptions,
        use_symmetry_canonicalization: bool = False,
        n_distance_nodes: int = 8,
    ):
        self.model = model
        self.options = options
        self.device = torch.device(options.device)
        self.dtype = options.dtype
        self.use_symmetry_canonicalization = use_symmetry_canonicalization
        self.n_distance_nodes = int(n_distance_nodes)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        options: InferenceOptions | None = None,
    ) -> "FastCutCellInference":
        """Build optimized inference runtime from a saved checkpoint."""
        if options is None:
            options = InferenceOptions()

        if options.num_threads > 0:
            torch.set_num_threads(options.num_threads)
        if options.num_interop_threads > 0:
            try:
                torch.set_num_interop_threads(options.num_interop_threads)
            except RuntimeError as exc:
                warnings.warn(
                    f"Skipping torch.set_num_interop_threads after runtime init ({exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )

        ckpt = torch.load(checkpoint_path, map_location=options.device, weights_only=False)

        model_type = ckpt.get("model_type", "mlp_default")
        train_cfg = ckpt.get("train_cfg", ckpt.get("train_config", {})) or {}

        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
        if state_dict is None:
            raise KeyError("Checkpoint does not contain model_state_dict or state_dict")

        hidden = int(train_cfg.get("hidden", 256))
        dropout_p = float(train_cfg.get("dropout_p", 0.0))
        model_cfg = ckpt.get("model_cfg", ckpt.get("model_config", {})) or {}
        include_nu = bool(model_cfg.get("include_nu_input", True))
        n_distance_nodes = int(model_cfg.get("n_distance_nodes", 8))
        if "n_distance_nodes" not in model_cfg and "pos_embed" in state_dict:
            n_distance_nodes = int(state_dict["pos_embed"].shape[0])
        if n_distance_nodes > 8:
            feature_order = ckpt.get("sdf_feature_order")
            if feature_order is None:
                warnings.warn(
                    "Rich-SDF checkpoint has no feature-order metadata; assuming the current "
                    f"{REFINED_SDF_FEATURE_ORDER!r} convention from its positional embedding.",
                    RuntimeWarning,
                )
            elif feature_order != REFINED_SDF_FEATURE_ORDER:
                raise ValueError(
                    "Refined checkpoint SDF feature order is missing or incompatible: "
                    f"expected {REFINED_SDF_FEATURE_ORDER!r}, got {feature_order!r}. "
                    "Use a checkpoint trained with the current feature order, or migrate a "
                    "legacy checkpoint with clean_simulation/scripts/"
                    "migrate_refined_checkpoint_sdf_order.py."
                )
        if "input_proj.weight" in state_dict:
            hidden = int(state_dict["input_proj.weight"].shape[0])
            include_nu = bool(state_dict["input_proj.weight"].shape[1] == 9) if "include_nu_input" not in model_cfg else include_nu
        if "hidden_layers.0.weight" in state_dict or model_type == "simple_relu":
            raise ValueError(
                "Legacy simple_relu checkpoints are no longer supported. "
                "Use a residual CutCellNet or transformer checkpoint."
            )
        elif model_type in ("transformer",) or "node_embed.weight" in state_dict:
            from clean_simulation.internal_nn_runtime.cutcell_ml.model import CutCellNetTransformer
            embed_dim   = int(state_dict["node_embed.weight"].shape[0]) if "node_embed.weight" in state_dict else int(train_cfg.get("fusion_dim", 128))
            # Count transformer layers from state_dict
            n_layers    = sum(1 for k in state_dict if k.startswith("transformer.layers.") and k.endswith(".self_attn.in_proj_weight"))
            if n_layers == 0:
                n_layers = int(train_cfg.get("n_residual_blocks", 4))
            n_heads     = int(train_cfg.get("n_heads", 8 if embed_dim % 8 == 0 else 4))
            ffn_weight = state_dict.get("transformer.layers.0.linear1.weight")
            ffn_dim = int(ffn_weight.shape[0]) if ffn_weight is not None else 4 * embed_dim
            dropout_p_t = float(train_cfg.get("dropout_p", 0.0))
            model = CutCellNetTransformer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                ffn_dim=ffn_dim,
                dropout=dropout_p_t,
                ordered_sdf_branch="ordered_sdf_branch.0.weight" in state_dict,
                n_distance_nodes=n_distance_nodes,
            )
        else:
            # Always derive fusion_dim from the actual weight shape rather than
            # train_cfg, which may be stale if the model was retrained at a
            # different width without updating the config snapshot.
            fusion_dim = (
                int(state_dict["input_proj.weight"].shape[0])
                if "input_proj.weight" in state_dict
                else int(train_cfg.get("fusion_dim", hidden))
            )
            # Derive n_residual_blocks from the state_dict itself so the model
            # shape always matches the checkpoint regardless of train_cfg.
            trunk_indices = [int(k.split('.')[1]) for k in state_dict if k.startswith('trunk.')]
            n_residual_blocks = (max(trunk_indices) + 1) if trunk_indices else int(train_cfg.get("n_residual_blocks", 4))
            use_class_conditioning = bool(train_cfg.get("use_class_conditioning", False))
            class_encoder_type = str(train_cfg.get("class_encoder_type", "one_hot"))
            class_embed_dim = int(train_cfg.get("class_embed_dim", 32)) if use_class_conditioning else 0
            include_class_inputs = bool(train_cfg.get("include_class_inputs", False))
            trunk_type = str(model_cfg.get("trunk_type", train_cfg.get("trunk_type", "resnet")))
            
            # Derive residual scaling from checkpoint tensors to preserve weight compatibility.
            has_alpha = any(k.endswith(".alpha") for k in state_dict.keys())
            if has_alpha:
                residual_scaling = "rezero"
            else:
                has_trunk = any(k.startswith("trunk.") for k in state_dict.keys())
                if has_trunk:
                    residual_scaling = "none"
                else:
                    residual_scaling = "rezero"

            if "input_proj.weight" in state_dict:
                actual_in_dim = state_dict["input_proj.weight"].shape[1]
                expected_base = n_distance_nodes + (1 if include_nu else 0)
                if include_class_inputs:
                    expected_base += 14
                if actual_in_dim == expected_base + 15:
                    use_class_conditioning = True
                    class_encoder_type = "one_hot"
                    class_embed_dim = 15

            # Derive matrix_size and use_reduced_null_space from the
            # checkpoint's output_proj shape, so the model always matches
            # regardless of defaults.
            # N_OUT = matrix_size*(matrix_size+1)//2  →  solve the quadratic.
            # Old checkpoints (24×24 full upper-tri) → n_out=300 → matrix_size=24
            # New checkpoints (18×18 reduced null-space) → n_out=171 → matrix_size=18
            use_reduced_null_space = bool(train_cfg.get("use_reduced_null_space", True))
            if "output_proj.weight" in state_dict:
                n_out = int(state_dict["output_proj.weight"].shape[0])
                import math
                matrix_size = int((-1 + math.isqrt(1 + 8 * n_out)) // 2)
                # If n_out=300 (old 24×24 checkpoint), disable null-space reduction
                if n_out == 300:
                    use_reduced_null_space = False
            else:
                matrix_size = int(train_cfg.get("matrix_size", CutCellNet.MATRIX_SIZE))

            model = CutCellNet(
                fusion_dim=fusion_dim,
                n_residual_blocks=n_residual_blocks,
                matrix_size=matrix_size,
                class_embed_dim=class_embed_dim,
                class_encoder_type=class_encoder_type,
                dropout_p=dropout_p,
                include_nu=include_nu,
                use_class_conditioning=use_class_conditioning,
                include_class_inputs=include_class_inputs,
                trunk_type=trunk_type,
                residual_scaling=residual_scaling,
                use_reduced_null_space=use_reduced_null_space,
                n_distance_nodes=n_distance_nodes,
            )
        # Use strict=False so newly added non-parameter buffers do not break
        # loading checkpoints that were saved before those buffers existed.
        # Normalise k_scale shape: old checkpoints store it as scalar [],
        # current model registers it as shape [1].
        if "k_scale" in state_dict and state_dict["k_scale"].shape == torch.Size([]):
            state_dict["k_scale"] = state_dict["k_scale"].unsqueeze(0)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            warnings.warn(
                f"Checkpoint has unexpected keys (ignored): {incompatible.unexpected_keys}"
            )
        model.eval()

        normalizers = ckpt.get("normalizers", {}) or {}
        if hasattr(model, "set_normalizers") and normalizers:
            in_mean = torch.as_tensor(normalizers.get("in_mean", []), dtype=options.dtype)
            in_std = torch.as_tensor(normalizers.get("in_std", []), dtype=options.dtype).clamp(min=1e-6)
            k_scale = torch.as_tensor(normalizers.get("k_scale", 1.0), dtype=options.dtype).reshape(1)
            expected_input_dim = n_distance_nodes + (1 if include_nu else 0)
            if in_mean.numel() == expected_input_dim and in_std.numel() == expected_input_dim:
                model.set_normalizers(in_mean, in_std, k_scale)

        model = model.to(device=options.device, dtype=options.dtype)
        wrapped: nn.Module = _BatchedWrapper(model).eval()

        compiled_ok = False
        if options.use_compile and hasattr(torch, "compile"):
            # torch.compile does not accept the ScriptModule produced by JIT tracing here, so the
            # compiled path takes precedence and stays on the eager batched wrapper.
            try:
                wrapped = torch.compile(wrapped, mode=options.compile_mode, fullgraph=False, dynamic=False)
                compiled_ok = True
            except Exception as exc:
                warnings.warn(
                    f"torch.compile unavailable for this runtime ({type(exc).__name__}: {exc}); "
                    "falling back to JIT optimization.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if not compiled_ok and options.use_jit:
            ex_d = torch.zeros(
                (max(1, options.warmup_batch_size), n_distance_nodes),
                dtype=options.dtype,
                device=options.device,
            )
            ex_nu = torch.zeros((max(1, options.warmup_batch_size),), dtype=options.dtype, device=options.device)
            # Disable check_trace for Transformer models: the JIT mangles the class
            # name on the second invocation, causing a spurious TracingCheckError.
            is_transformer = isinstance(getattr(wrapped, "base", None),
                                        __import__("clean_simulation.internal_nn_runtime.cutcell_ml.model",
                                                   fromlist=["CutCellNetTransformer"]).CutCellNetTransformer)
            traced = torch.jit.trace(wrapped, (ex_d, ex_nu), strict=False,
                                     check_trace=not is_transformer)
            traced = torch.jit.freeze(traced.eval())
            wrapped = torch.jit.optimize_for_inference(traced)

        use_sym = bool(train_cfg.get("use_symmetry_canonicalization", False))
        engine = cls(
            wrapped.eval(),
            options,
            use_symmetry_canonicalization=use_sym,
            n_distance_nodes=n_distance_nodes,
        )
        try:
            engine.warmup()
        except Exception as exc:
            if compiled_ok and options.use_jit:
                warnings.warn(
                    f"Compiled runtime warmup failed ({type(exc).__name__}: {exc}); "
                    "falling back to JIT optimization.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                ex_d = torch.zeros(
                    (max(1, options.warmup_batch_size), n_distance_nodes),
                    dtype=options.dtype,
                    device=options.device,
                )
                ex_nu = torch.zeros((max(1, options.warmup_batch_size),), dtype=options.dtype, device=options.device)
                traced = torch.jit.trace(_BatchedWrapper(model).eval(), (ex_d, ex_nu), strict=False)
                traced = torch.jit.freeze(traced.eval())
                wrapped = torch.jit.optimize_for_inference(traced)
                engine = cls(
                    wrapped.eval(),
                    options,
                    use_symmetry_canonicalization=use_sym,
                    n_distance_nodes=n_distance_nodes,
                )
                engine.warmup()
            else:
                raise

        return engine

    @classmethod
    def from_config_and_checkpoint(
        cls,
        config_path: Path,
        checkpoint_path: Path,
        options: InferenceOptions | None = None,
    ) -> "FastCutCellInference":
        """Build runtime using explicit JSON config with checkpoint fallback."""
        _ = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls.from_checkpoint(checkpoint_path=checkpoint_path, options=options)

    def warmup(self) -> None:
        """Run warmup passes to stabilize runtime performance."""
        if self.options.warmup_iters <= 0:
            return

        bs = max(1, int(self.options.warmup_batch_size))
        d = torch.zeros((bs, self.n_distance_nodes), dtype=self.dtype, device=self.device)
        nu = torch.zeros((bs,), dtype=self.dtype, device=self.device)

        with torch.inference_mode():
            for _ in range(self.options.warmup_iters):
                K_unit, log_norm = self.model(d, nu)
                _ = K_unit * torch.exp(log_norm)[:, None, None]

    @torch.inference_mode()
    def predict_torch(self, d: torch.Tensor, nu: torch.Tensor | None = None) -> torch.Tensor:
        """Fast inference from torch tensors. Returns reconstructed physical K."""
        d_in = d.to(device=self.device, dtype=self.dtype, non_blocking=False)
        nu_in = None if nu is None else nu.to(device=self.device, dtype=self.dtype, non_blocking=False)
        K_unit, log_norm = self.model(d_in, nu_in)
        if K_unit.ndim == 2:
            return K_unit * torch.exp(log_norm)
        return K_unit * torch.exp(log_norm)[:, None, None]

    @torch.inference_mode()
    def predict_numpy(self, d: Any, nu: Any = None) -> Any:
        """Fast inference from numpy-like arrays, returns numpy array of physical K."""
        d_t = torch.as_tensor(d, dtype=self.dtype, device=self.device)
        nu_t = None if nu is None else torch.as_tensor(nu, dtype=self.dtype, device=self.device)
        K_unit, log_norm = self.model(d_t, nu_t)
        if K_unit.ndim == 2:
            out = K_unit * torch.exp(log_norm)
        else:
            out = K_unit * torch.exp(log_norm)[:, None, None]
        return out.detach().cpu().numpy()
