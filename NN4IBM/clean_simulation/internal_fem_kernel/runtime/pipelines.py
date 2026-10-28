from __future__ import annotations

import time
from typing import Any, Dict

import torch

from ..fem import FEM
from .assembly_ops import assemble_force_base, assemble_stiffness_base
from .common import as_bool_mask, as_index_tensor_cpu, finalize_pipeline_timings, reset_pipeline_runtime_caches, sync_cuda_if_needed
from clean_simulation.internal_nn_runtime.geometry import (
    compute_nn_cutcell_stiffness,
    prepare_nn_inputs,
)
from .solver import solve_with_nitsche


def run_queso_pipeline_timed(
    grid,
    material,
    config,
    ips,
    top_mesh,
    poly_order,
    g_order,
    device="cuda",
) -> Dict[str, object]:
    timings: Dict[str, float] = {}
    dev = torch.device(device)
    reset_pipeline_runtime_caches(dev)

    timings["classification"] = 0.0

    print("[PIPELINE QUESO] Starting stiffness computation from QuESo points...")
    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    ke_list, active_indices = FEM.StiffnessAssembly.compute_element_matrices_from_queso_points(
        grid=grid,
        material=material,
        ips=ips,
        poly_order=poly_order,
        device=device,
    )

    sync_cuda_if_needed(dev)
    t_stiff = time.perf_counter() - t0
    timings["stiffness_inner"] = t_stiff
    timings["stiffness_cut"] = 0.0
    print(f"[PIPELINE QUESO] Stiffness computation finished in {t_stiff:.4f} seconds.")

    active_indices_t = active_indices.to(dtype=torch.int64, device=device)

    print("[PIPELINE QUESO] Assembling global stiffness matrix...")
    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    k_base = assemble_stiffness_base(
        ke_list=ke_list,
        active_indices=active_indices_t,
        grid=grid,
        poly_order=poly_order,
        device=device,
    )

    sync_cuda_if_needed(dev)
    t_stiff_asm = time.perf_counter() - t0
    timings["stiffness_assembly"] = t_stiff_asm
    print(f"[PIPELINE QUESO] Global stiffness assembly finished in {t_stiff_asm:.4f} seconds.")

    print("[PIPELINE QUESO] Assembling global force vector...")
    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()

    f_base = assemble_force_base(
        grid=grid,
        active_indices=active_indices_t,
        ips=ips,
        config=config,
        poly_order=poly_order,
    )

    sync_cuda_if_needed(dev)
    t_force_asm = time.perf_counter() - t0
    timings["force_assembly"] = t_force_asm
    print(f"[PIPELINE QUESO] Global force assembly finished in {t_force_asm:.4f} seconds.")

    print("[PIPELINE QUESO] Solving system with Nitsche BCs...")
    u_nodes, u_vec, n_nodes_total, solve_timings, _, k_global = solve_with_nitsche(
        grid=grid,
        material=material,
        config=config,
        top_mesh=top_mesh,
        active_indices=active_indices_t,
        poly_order=poly_order,
        g_order=g_order,
        k_base=k_base,
        f_base=f_base,
        device=device,
    )

    timings.update(solve_timings)
    timings = finalize_pipeline_timings(timings)
    print(f"[PIPELINE QUESO] Total pipeline time: {timings.get('total', 0.0):.4f} seconds.")
    return {
        "u_nodes": u_nodes,
        "u_vec": u_vec,
        "n_nodes_total": n_nodes_total,
        "active_indices": active_indices_t,
        "timings": timings,
        "k_global": k_global,
    }


def run_classic_pipeline_timed(
    grid,
    volume,
    material,
    config,
    ips_for_body_force,
    top_mesh,
    poly_order,
    g_order,
    order=2,
    cut_max_depth=3,
    device="cuda",
    lower_threshold=None,
    upper_threshold=None,
) -> Dict[str, object]:
    timings: Dict[str, float] = {}
    dev = torch.device(device)
    reset_pipeline_runtime_caches(dev)

    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    inside, partial, _ = FEM.Classification.classify_sdf(
        volume=volume,
        grid=grid,
        device=device,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )
    sync_cuda_if_needed(dev)
    timings["classification"] = time.perf_counter() - t0

    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    ke_list, active_indices, stiff_diag = FEM.StiffnessAssembly.compute_element_matrices(
        grid=grid,
        volume=volume,
        material=material,
        inside=inside,
        partial=partial,
        order=order,
        cut_max_depth=int(cut_max_depth),
        device=device,
        poly_order=poly_order,
        return_diagnostics=True,
    )
    sync_cuda_if_needed(dev)
    t_stiffness_stage = time.perf_counter() - t0

    timings["stiffness_cut"] = (
        float(stiff_diag.get("partial_coords_build", 0.0))
        + float(stiff_diag.get("partial_compute", 0.0))
        + float(stiff_diag.get("norm_filter", 0.0))
    )
    timings["stiffness_inner"] = float(max(t_stiffness_stage - timings["stiffness_cut"], 0.0))

    active_indices_t = active_indices.to(dtype=torch.int64, device=device)

    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    k_base = assemble_stiffness_base(
        ke_list=ke_list,
        active_indices=active_indices_t,
        grid=grid,
        poly_order=poly_order,
        device=device,
    )
    sync_cuda_if_needed(dev)
    timings["stiffness_assembly"] = time.perf_counter() - t0

    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    f_base = assemble_force_base(
        grid=grid,
        active_indices=active_indices_t,
        ips=ips_for_body_force,
        config=config,
        poly_order=poly_order,
    )
    sync_cuda_if_needed(dev)
    timings["force_assembly"] = time.perf_counter() - t0

    u_nodes, u_vec, n_nodes_total, solve_timings, nitsche_sets, k_global = solve_with_nitsche(
        grid=grid,
        material=material,
        config=config,
        top_mesh=top_mesh,
        active_indices=active_indices_t,
        poly_order=poly_order,
        g_order=g_order,
        k_base=k_base,
        f_base=f_base,
        device=device,
    )
    timings.update(solve_timings)
    timings = finalize_pipeline_timings(timings)
    return {
        "u_nodes": u_nodes,
        "u_vec": u_vec,
        "n_nodes_total": n_nodes_total,
        "active_indices": active_indices_t,
        "nitsche_sets": nitsche_sets,
        "timings": timings,
        "k_global": k_global,
    }


def run_hybrid_pipeline_timed(
    grid,
    volume,
    material,
    config,
    ips_for_body_force,
    top_mesh,
    nn_engine,
    poly_order,
    g_order,
    order=2,
    classic_nitsche_sets=None,
    active_indices_reference=None,
    device="cuda",
    lower_threshold=None,
    upper_threshold=None,
) -> Dict[str, object]:
    timings: Dict[str, float] = {}
    dev = torch.device(device)
    reset_pipeline_runtime_caches(dev)

    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    inside, partial, _ = FEM.Classification.classify_sdf(
        volume=volume,
        grid=grid,
        device=device,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )
    inside = as_bool_mask(inside, device=device)
    partial = as_bool_mask(partial, device=device)
    sync_cuda_if_needed(dev)
    timings["classification"] = time.perf_counter() - t0

    inside_only = inside.reshape(-1).clone()

    ref_mask = None
    if active_indices_reference is not None:
        active_ref = active_indices_reference
        ref_mask = torch.zeros(int(grid.shape[0] * grid.shape[1] * grid.shape[2]), dtype=torch.bool, device=device)
        ref_mask[active_ref] = True
        inside_only &= ref_mask

    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    ke_inside, active_inside, _ = FEM.StiffnessAssembly.compute_inside_element_matrices(
        grid=grid,
        material=material,
        inside=inside_only,
        order=order,
        device=device,
        poly_order=poly_order,
        return_diagnostics=True,
    )
    sync_cuda_if_needed(dev)
    timings["stiffness_inner"] = time.perf_counter() - t0

    partial_mask = partial.reshape(-1).clone()
    if ref_mask is not None:
        partial_mask &= ref_mask
    partial_indices = torch.nonzero(partial_mask, as_tuple=False).reshape(-1).to(dtype=torch.int64, device=device)

    sync_cuda_if_needed(dev)
    t_cut = time.perf_counter()
    d_cut, nu_cut, h_ref = prepare_nn_inputs(
        grid=grid,
        volume=volume,
        partial_indices=partial_indices,
        material=material,
        poly_order=poly_order,
    )

    print("\n===== INFERENCE FEATURES =====")
    for i in range(min(10, len(partial_indices))):
        print(f"row {i:3d} flat={int(partial_indices[i])}")
        print(d_cut[i].numpy())
    print("==============================\n")


    
    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    ke_cut, active_cut = compute_nn_cutcell_stiffness(
        d_cut=d_cut,
        nu_cut=nu_cut,
        nn_engine=nn_engine,
        material=material,
        partial_indices=partial_indices,
        h_ref=h_ref,
    )
    sync_cuda_if_needed(dev)
    timings["diag_stiffness_cut_infer"] = time.perf_counter() - t0
    timings["stiffness_cut"] = time.perf_counter() - t_cut

    active_inside_t = active_inside.to(dtype=torch.int64, device=device)
    ke_list = list(ke_inside) + list(ke_cut)
    active_indices = torch.cat([active_inside_t, active_cut.to(dtype=torch.int64, device=device)], dim=0)

    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    k_base = assemble_stiffness_base(
        ke_list=ke_list,
        active_indices=active_indices,
        grid=grid,
        poly_order=poly_order,
        device=device,
    )
    sync_cuda_if_needed(dev)
    timings["stiffness_assembly"] = time.perf_counter() - t0

    sync_cuda_if_needed(dev)
    t0 = time.perf_counter()
    f_base = assemble_force_base(
        grid=grid,
        active_indices=active_indices,
        ips=ips_for_body_force,
        config=config,
        poly_order=poly_order,
    )
    sync_cuda_if_needed(dev)
    timings["force_assembly"] = time.perf_counter() - t0

    u_nodes, u_vec, n_nodes_total, solve_timings, _, k_global = solve_with_nitsche(
        grid=grid,
        material=material,
        config=config,
        top_mesh=top_mesh,
        active_indices=active_indices,
        poly_order=poly_order,
        g_order=g_order,
        k_base=k_base,
        f_base=f_base,
        precomputed_nitsche_sets=classic_nitsche_sets,
        device=device,
    )
    timings.update(solve_timings)
    timings = finalize_pipeline_timings(timings)
    return {
        "u_nodes": u_nodes,
        "u_vec": u_vec,
        "n_nodes_total": n_nodes_total,
        "active_indices": active_indices,
        "inside_count": int(active_inside_t.numel()),
        "partial_count": int(active_cut.numel()),
        "timings": timings,
        "k_global": k_global,
    }

