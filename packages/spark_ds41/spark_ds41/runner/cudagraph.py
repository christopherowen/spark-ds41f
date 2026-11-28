# SPDX-License-Identifier: Apache-2.0
"""CUDA graph node-priority instantiation helper for GB10."""
from __future__ import annotations

import ctypes
import torch

_CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 0x08
_CUDART = None


def _get_cudart():
    global _CUDART
    if _CUDART is None:
        lib = ctypes.CDLL("libcudart.so")
        lib.cudaGraphInstantiateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        lib.cudaGraphInstantiateWithFlags.restype = ctypes.c_int
        lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cudaGraphLaunch.restype = ctypes.c_int
        lib.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
        lib.cudaGraphExecDestroy.restype = ctypes.c_int
        _CUDART = lib
    return _CUDART


class _PriorityCUDAGraphExec:
    def __init__(self, graph: torch.cuda.CUDAGraph, exec_ptr: ctypes.c_void_p, lib):
        self._graph = graph
        self._exec_ptr = exec_ptr
        self._lib = lib

    def replay(self) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        rc = self._lib.cudaGraphLaunch(self._exec_ptr, ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"cudaGraphLaunch failed with error {rc}")

    def __del__(self) -> None:
        exec_ptr = getattr(self, "_exec_ptr", None)
        if exec_ptr and exec_ptr.value:
            try:
                self._lib.cudaGraphExecDestroy(exec_ptr)
            except Exception:
                pass
            self._exec_ptr = ctypes.c_void_p()

    def __getattr__(self, name: str):
        return getattr(self._graph, name)


def instantiate_priority_graph(graph: torch.cuda.CUDAGraph):
    try:
        raw_fn = getattr(graph, "raw_cuda_graph", None)
        if raw_fn is None:
            return graph
        raw = raw_fn()
        if not raw:
            return graph
        lib = _get_cudart()
        exec_ptr = ctypes.c_void_p()
        rc = lib.cudaGraphInstantiateWithFlags(
            ctypes.byref(exec_ptr),
            ctypes.c_void_p(raw),
            ctypes.c_uint64(_CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY),
        )
        if rc != 0 or not exec_ptr.value:
            return graph
        return _PriorityCUDAGraphExec(graph, exec_ptr, lib)
    except Exception:
        return graph
