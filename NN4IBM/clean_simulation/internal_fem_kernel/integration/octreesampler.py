import torch
import hashlib
import numpy as np
from shape import Shape


_CO = torch.tensor([
    [-1., -1., -1.], [ 1., -1., -1.], [ 1.,  1., -1.], [-1.,  1., -1.],
    [-1., -1.,  1.], [ 1., -1.,  1.], [ 1.,  1.,  1.], [-1.,  1.,  1.],
], dtype=torch.float64)


class OctreeSampler:

    _last_cache = None

    # Cache of 1D Gauss-Legendre rules on [-1,1], keyed by n_pts.
    _gauss_1d_cache = {}

    @staticmethod
    def _cache_key(coords_batch, volume, n_pts, device, max_depth, poly_order, compute_physical):
        coords_cpu = coords_batch.detach().cpu().contiguous().numpy()
        digest = hashlib.blake2b(coords_cpu.tobytes(), digest_size=16).hexdigest()

        # Fingerprint geometric state rather than Python object identity.
        # Object ids can be reused after deallocation, which can cause stale cache hits.
        vol_state = []
        for attr in ("node_coords", "nodal_sdf", "xyz_min", "scale"):
            if hasattr(volume, attr):
                arr = np.asarray(getattr(volume, attr), dtype=np.float64)
                vol_state.append((attr, arr.shape, hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()))
        vol_signature = tuple(vol_state)

        return (
            vol_signature,
            int(n_pts),
            str(device),
            int(max_depth),
            int(poly_order),
            bool(compute_physical),
            tuple(coords_cpu.shape),
            digest,
        )

    @staticmethod
    def _gauss_leaf_points(n_pts, device, dtype):
        """
        Tensor-product Gauss-Legendre rule on the reference cube [-1,1]^3
        with n_pts points per direction (n_pts**3 points total).

        Returns:
            pts  : (n_pts**3, 3) local coordinates in [-1,1]^3
            wgts : (n_pts**3,)   weights normalized so that they sum to 8.0
                   (the reference-volume of [-1,1]^3), matching the
                   convention used by the rest of the sampler where a
                   subcube's quadrature weights integrate to its reference
                   volume (2*half)**3.
        """
        key = (int(n_pts), str(device), str(dtype))
        cached = OctreeSampler._gauss_1d_cache.get(key)
        if cached is None:
            nodes_1d, w_1d = np.polynomial.legendre.leggauss(int(n_pts ** (1/3)))  # n_pts**(1/3) points per direction
            nodes_1d_t = torch.as_tensor(nodes_1d, dtype=dtype, device=device)
            w_1d_t = torch.as_tensor(w_1d, dtype=dtype, device=device)

            # tensor product over x, y, z
            gx, gy, gz = torch.meshgrid(nodes_1d_t, nodes_1d_t, nodes_1d_t, indexing='ij')
            pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)  # (n_pts**3, 3)

            wx, wy, wz = torch.meshgrid(w_1d_t, w_1d_t, w_1d_t, indexing='ij')
            wgts = (wx * wy * wz).reshape(-1)  # (n_pts**3,), sums to 8.0 (2*2*2)

            cached = (pts, wgts)
            OctreeSampler._gauss_1d_cache[key] = cached

        return cached

    @staticmethod
    def sample_batch(coords_batch, volume, n_pts,
                     device='cpu', max_depth=4, cut_max_depth=4, poly_order=1,
                     compute_physical=True,
                     debug=False, debug_prefix=''):

        dtype = torch.float64
        coords_batch = torch.as_tensor(coords_batch, dtype=dtype, device=device)

        E = coords_batch.shape[0]
        n_nodes = coords_batch.shape[1]
        geom_poly_order = Shape.infer_poly_order_from_nnodes(n_nodes)

        # Per-leaf Gauss-Legendre rule, n_pts points per direction.
        # local_pts are coordinates in [-1,1]^3, local_w are weights summing
        # to 8.0 (the reference volume of [-1,1]^3).
        local_pts, local_w = OctreeSampler._gauss_leaf_points(n_pts, device, dtype)
        n_leaf_pts = local_pts.shape[0]  # n_pts**3

        # -------------------------------------------------
        # INIT ROOT CELLS
        # -------------------------------------------------
        centres = torch.zeros(E, 1, 3, dtype=dtype, device=device)  # (E,1,3)
        elem_ids = torch.arange(E, device=device).view(E, 1)        # (E,1)

        half = 1.0

        # final storage (flat) — one entry per (accepted leaf x n_leaf_pts)
        pts_ref_all = []
        w_all       = []
        h_all       = []
        eid_all     = []

        collected = torch.zeros(E, dtype=torch.long, device=device)
        level_stats = []

        co = _CO.to(device=device, dtype=dtype)  # (8,3)

        # -------------------------------------------------
        # OCTREE LOOP
        # -------------------------------------------------
        # ``max_depth`` is the number of actual subdivisions performed, matching
        # FCMLab's ``refinement_depth`` convention.  Iteration range is
        # [0 .. max_depth] inclusive: levels 0..max_depth-1 subdivide cut cubes;
        # level max_depth is the terminal level where cut subcubes are accepted.
        #
        # ``cut_max_depth`` independently caps how deep *cut* cells may keep
        # subdividing. Once a branch has been classified as "cut" for
        # cut_max_depth consecutive subdivisions without resolving to fully
        # inside/outside, it is forced to terminate (treated as if it were
        # the terminal level for that branch) rather than refining further.
        # This lets callers ask for a coarser geometric cut-off than the
        # general max_depth, e.g. to bound cost on very rough surfaces.
        effective_max_depth = max(int(max_depth), int(cut_max_depth))

        for level in range(effective_max_depth + 1):

            # flatten cubes
            centres_flat = centres.reshape(-1, 3)
            elem_flat    = elem_ids.reshape(-1)

            N3_cubes = centres_flat.shape[0]

            # ---------------------------------------------
            # CORNERS (reference)
            # ---------------------------------------------
            corners_ref = (
                centres_flat[:, None, :] + co[None, :, :] * half
            )  # (N3c,8,3)

            # ---------------------------------------------
            # MAP TO PHYSICAL
            # ---------------------------------------------
            corners_flat = corners_ref.reshape(-1, 3)

            N3vals = Shape.N3(
                corners_flat[:, 0],
                corners_flat[:, 1],
                corners_flat[:, 2],
                poly_order=geom_poly_order,
            ).to(dtype).view(N3_cubes, 8, n_nodes)

            parent_coords = coords_batch[elem_flat]

            corners_phys = torch.einsum(
                'nij,njk->nik',
                N3vals,
                parent_coords
            )  # (N3c,8,3)

            # ---------------------------------------------
            # SDF (single call)
            # ---------------------------------------------
            sdf = volume.signed_distance_torch(
                corners_phys.reshape(-1, 3)
            ).view(N3_cubes, 8)

            # ---------------------------------------------
            # CLASSIFICATION
            # ---------------------------------------------
            # Robust sign classification near the surface. Small numerical
            # jitter around 0 can break mirror symmetry for cut cells.
            eps = 1e-10

            inside  = (sdf >  eps).all(dim=1).to(device)
            outside = (sdf < -eps).all(dim=1).to(device)
            cut     = ~(inside | outside).to(device)

            inside_count = int(inside.sum().item())
            outside_count = int(outside.sum().item())
            cut_count = int(cut.sum().item())

            # ---------------------------------------------
            # ACCEPTANCE RULE
            # ---------------------------------------------
            # FCMLab-compatible acceptance rule:
            #   • At non-terminal levels: accept fully-inside cubes.
            #   • At the terminal level : accept inside cubes and only those
            #     cut cubes whose center lies inside the physical domain.
            # A level is terminal either because we hit the global
            # effective_max_depth, or because this cube's branch has already
            # used up its cut-cell subdivision budget (cut_max_depth).
            is_last_level = (level == effective_max_depth)

            if is_last_level:
                force_terminal_cut = cut
            else:
                # cubes that are cut and have reached their own cut-depth
                # budget are also terminated early, even though other
                # branches may keep subdividing.
                force_terminal_cut = cut & (level >= int(cut_max_depth))

            is_terminal_for_cube = is_last_level | (~cut) | force_terminal_cut
            # ^ inside/outside cubes are always "terminal" in the sense that
            # they don't subdivide further; only cut cubes choose to keep
            # going unless force_terminal_cut applies.

            if is_last_level:
                terminal_cut_now = cut
            else:
                terminal_cut_now = force_terminal_cut

            if terminal_cut_now.any() or inside.any():
                # Map subcell centers from reference to physical and keep cut
                # cubes only when center-inside. This matches FCMLab center
                # point material-selection semantics used in cut-cell tests.
                N3c_ctr = Shape.N3(
                    centres_flat[:, 0],
                    centres_flat[:, 1],
                    centres_flat[:, 2],
                    poly_order=geom_poly_order,
                ).to(dtype)
                ctr_phys = (N3c_ctr.unsqueeze(-1) * parent_coords).sum(dim=1)
                sdf_ctr = volume.signed_distance_torch(ctr_phys).view(N3_cubes)
                center_inside = sdf_ctr > eps
            else:
                center_inside = torch.zeros(N3_cubes, dtype=torch.bool, device=device)
            center_inside = center_inside.to(device)

            accept = inside | (terminal_cut_now & center_inside)

            accept_count = int(accept.sum().item())

            level_stats.append({
                "level": int(level),
                "half": float(half),
                "n_cubes": int(N3_cubes),
                "inside": inside_count,
                "outside": outside_count,
                "cut": cut_count,
                "accepted": accept_count,
            })

            if debug:
                prefix = f"{debug_prefix} " if debug_prefix else ""

            # ---------------------------------------------
            # STORE ACCEPTED CUBES — n_pts**3 quadrature points each
            # ---------------------------------------------
            if accept.any():

                acc_centres = centres_flat[accept]   # (Na,3)
                acc_eids    = elem_flat[accept]       # (Na,)
                Na = acc_centres.shape[0]

                # Map each leaf's local Gauss points (in [-1,1]^3) into the
                # leaf's own reference frame: x = centre + half * local.
                # (Na, n_leaf_pts, 3)

                pts_ref_leaf = (
                    acc_centres[:, None, :]
                    + half * local_pts[None, :, :]
                )

                # Quadrature weight for each point: local_w integrates over
                # [-1,1]^3 (volume 8); scale by (half**3) so the points for
                # this leaf integrate to the leaf's true reference volume
                # (2*half)**3, consistent with the previous single-point
                # convention of weight = (2*half)**3 / 8 per accepted cube
                # (n_pts=1 with a unit weight of 8 recovers that exactly).
                w_leaf = (half ** 3) * local_w  # (n_leaf_pts,)
                w_leaf_full = w_leaf[None, :].expand(Na, n_leaf_pts)  # (Na, n_leaf_pts)

                pts_ref_all.append(pts_ref_leaf.reshape(Na * n_leaf_pts, 3))
                w_all.append(w_leaf_full.reshape(Na * n_leaf_pts))
                h_all.append(
                    torch.full((Na * n_leaf_pts,), half, dtype=dtype, device=device)
                )
                eid_all.append(acc_eids.repeat_interleave(n_leaf_pts))

                # update counters per element (count quadrature points, not cubes)
                collected.index_add_(
                    0, acc_eids,
                    torch.full_like(acc_eids, n_leaf_pts)
                )

            # ---------------------------------------------
            # TERMINATION / SUBDIVISION
            # ---------------------------------------------
            # At the last level there are no more subdivisions at all.
            if is_last_level:
                break

            # Only cubes that are cut AND have not been force-terminated
            # by the per-branch cut_max_depth budget continue subdividing.
            subdivide = cut & (~force_terminal_cut)

            if not subdivide.any():
                break

            centres_cut = centres_flat[subdivide]
            elem_cut    = elem_flat[subdivide]

            N3c = centres_cut.shape[0]

            # generate children
            child_half = half * 0.5

            offsets = co * child_half  # (8,3)

            centres = (
                centres_cut[:, None, :] + offsets[None, :, :]
            ).reshape(N3c * 8, 3)

            elem_ids = elem_cut.repeat_interleave(8)

            # reshape back to (E?, N3cubes?, 3) style (flat storage)
            centres = centres.unsqueeze(1)
            elem_ids = elem_ids.unsqueeze(1)

            half = child_half

        # -------------------------------------------------
        # FINAL ASSEMBLY
        # -------------------------------------------------
        if len(pts_ref_all) == 0:
            result = ([None]*E, [None]*E, None, [None]*E, [None]*E)
            return result

        pts_ref_all = torch.cat(pts_ref_all)
        w_all       = torch.cat(w_all)
        h_all       = torch.cat(h_all)
        eid_all     = torch.cat(eid_all)

        # sort by element
        order = torch.argsort(eid_all)

        pts_ref_all = pts_ref_all[order]
        w_all       = w_all[order]
        h_all       = h_all[order]
        eid_all     = eid_all[order]

        unique_eids, counts = torch.unique_consecutive(
            eid_all, return_counts=True
        )

        pts_ref_list  = [None] * E
        w_list        = [None] * E
        h_ref_list    = [None] * E
        pts_phys_list = [None] * E if compute_physical else None
        h_phys_list   = [None] * E if compute_physical else None

        if compute_physical:
            phys_extents = (
                coords_batch.max(dim=1).values -
                coords_batch.min(dim=1).values
            )
            mean_stretch = (phys_extents / 2.0).mean(dim=1)

        offset = 0

        for eid_t, cnt_t in zip(unique_eids, counts):

            eid = int(eid_t)
            cnt = int(cnt_t)

            pts_ref = pts_ref_all[offset:offset+cnt]
            weights = w_all[offset:offset+cnt]
            h_ref   = h_all[offset:offset+cnt]

            # Robustness: enforce normalized reference-volume weights per
            # element. Total reference volume of the parent cube is 8.0
            # (i.e. (2*1.0)**3 for the root [-1,1]^3 cube), so if accepted
            # leaves' quadrature weights sum to more than that, something
            # has gone numerically wrong (e.g. duplicate/overlapping
            # contributions) and weights are rescaled defensively.
            w_sum = torch.clamp(weights.sum(), min=torch.finfo(dtype).eps)
            if w_sum > 8.0:
                weights = weights * (8.0 / w_sum)

            if compute_physical:
                # map to physical only when requested by caller
                N3vals = Shape.N3(
                    pts_ref[:, 0],
                    pts_ref[:, 1],
                    pts_ref[:, 2],
                    poly_order=geom_poly_order,
                ).to(dtype)

                pts_phys = N3vals @ coords_batch[eid]

                h_phys = h_ref * mean_stretch[eid]

            pts_ref_list[eid]  = pts_ref
            w_list[eid]        = weights
            h_ref_list[eid]    = h_ref
            if compute_physical:
                pts_phys_list[eid] = pts_phys
                h_phys_list[eid]   = h_phys

            offset += cnt

        result = (
            pts_ref_list,
            w_list,
            h_ref_list,
            pts_phys_list,
            h_phys_list
        )
        return result