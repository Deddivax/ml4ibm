import numpy as np
import torch

from tensor import _tensor


GAUSS_1D_CACHE = {}
GAUSS_2D_CACHE = {}
GAUSS_3D_CACHE = {}
GAUSS_TRIANGLE_CACHE = {}
CACHE_ENABLED = False


def _cache_key(order, device, dtype):
    return int(order), str(device), str(dtype)


def _cache_enabled():
    return False


def gauss_1d(order, device, dtype=torch.float64):
    key = _cache_key(order, device, dtype)
    if _cache_enabled():
        cached = GAUSS_1D_CACHE.get(key)
        if cached is not None:
            return cached
    p1, w1 = np.polynomial.legendre.leggauss(order)
    out = (_tensor(p1, device, dtype), _tensor(w1, device, dtype))
    if _cache_enabled():
        GAUSS_1D_CACHE[key] = out
    return out


def gauss_3d(order, device, dtype=torch.float64):
    key = _cache_key(order, device, dtype)
    if _cache_enabled():
        cached = GAUSS_3D_CACHE.get(key)
        if cached is not None:
            return cached
    p1, w1 = np.polynomial.legendre.leggauss(order)
    xi, eta, zeta = np.meshgrid(p1, p1, p1, indexing="ij")
    wx, wy, wz = np.meshgrid(w1, w1, w1, indexing="ij")
    pts = np.stack([xi.ravel(), eta.ravel(), zeta.ravel()], axis=1)
    w = (wx * wy * wz).ravel()
    out = (_tensor(pts, device, dtype), _tensor(w, device, dtype))
    if _cache_enabled():
        GAUSS_3D_CACHE[key] = out
    return out


def gauss_2d(order, device="cpu", dtype=torch.float64):
    key = _cache_key(order, device, dtype)
    if _cache_enabled():
        cached = GAUSS_2D_CACHE.get(key)
        if cached is not None:
            return cached
    p, w = np.polynomial.legendre.leggauss(order)
    u, v = np.meshgrid(p, p, indexing="ij")
    wu, wv = np.meshgrid(w, w, indexing="ij")
    pts = np.stack([u.ravel(), v.ravel()], axis=1)
    weights = (wu * wv).ravel()
    out = (torch.tensor(pts, dtype=dtype, device=device), torch.tensor(weights, dtype=dtype, device=device))
    if _cache_enabled():
        GAUSS_2D_CACHE[key] = out
    return out


def gauss_triangle(order, device="cpu", dtype=torch.float64):
    key = _cache_key(order, device, dtype)
    if _cache_enabled():
        cached = GAUSS_TRIANGLE_CACHE.get(key)
        if cached is not None:
            return cached
    if order == 1:
        gp = torch.tensor([[1 / 3, 1 / 3]], dtype=dtype, device=device)
        gw = torch.tensor([0.5], dtype=dtype, device=device)
    elif order == 2:
        gp = torch.tensor([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], dtype=dtype, device=device)
        gw = torch.tensor([1 / 6, 1 / 6, 1 / 6], dtype=dtype, device=device)
    else:
        p, w = np.polynomial.legendre.leggauss(order)
        r = 0.5 * (p + 1.0)
        s = 0.5 * (p + 1.0)
        wr = 0.5 * w
        ws = 0.5 * w
        rr, ss = np.meshgrid(r, s, indexing="ij")
        wrr, wss = np.meshgrid(wr, ws, indexing="ij")
        u = rr.ravel()
        v = ((1.0 - rr) * ss).ravel()
        weights = (wrr * wss * (1.0 - rr)).ravel()
        gp = torch.tensor(np.stack([u, v], axis=1), dtype=dtype, device=device)
        gw = torch.tensor(weights, dtype=dtype, device=device)
    out = (gp, gw)
    if _cache_enabled():
        GAUSS_TRIANGLE_CACHE[key] = out
    return out


class Gauss:
    _gauss_1d_cache = GAUSS_1D_CACHE
    _gauss_2d_cache = GAUSS_2D_CACHE
    _gauss_3d_cache = GAUSS_3D_CACHE
    _gauss_triangle_cache = GAUSS_TRIANGLE_CACHE
    CACHE_ENABLED = False
    _cache_key = staticmethod(_cache_key)
    gauss_1d = staticmethod(gauss_1d)
    gauss_2d = staticmethod(gauss_2d)
    gauss_3d = staticmethod(gauss_3d)
    gauss_triangle = staticmethod(gauss_triangle)