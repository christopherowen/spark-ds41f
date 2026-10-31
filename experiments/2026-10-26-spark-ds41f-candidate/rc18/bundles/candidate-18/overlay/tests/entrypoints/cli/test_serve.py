# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``vllm serve`` CLI subcommand."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.cli.serve import run_headless


class _StopAfterEngineConfig(Exception):
    pass


def test_headless_imports_reasoning_parser_plugin_before_engine_config():
    plugin_path = "/tmp/custom_reasoning_parser.py"
    args = argparse.Namespace(
        api_server_count=0,
        reasoning_parser_plugin=plugin_path,
    )
    engine_args = MagicMock()

    with (
        patch(
            "vllm.entrypoints.cli.serve.vllm.AsyncEngineArgs.from_cli_args",
            return_value=engine_args,
        ),
        patch(
            "vllm.reasoning.ReasoningParserManager.import_reasoning_parser"
        ) as import_plugin,
    ):

        def create_engine_config(**kwargs):
            import_plugin.assert_called_once_with(plugin_path)
            raise _StopAfterEngineConfig

        engine_args.create_engine_config.side_effect = create_engine_config

        with pytest.raises(_StopAfterEngineConfig):
            run_headless(args)


def _make_follower_vllm_config():
    parallel_config = MagicMock()
    parallel_config.world_size = 3
    parallel_config.nnodes_within_dp = 3
    parallel_config.local_world_size = 1
    parallel_config.node_rank = 1
    parallel_config.node_rank_within_dp = 1
    parallel_config.tensor_parallel_size = 3
    parallel_config.pipeline_parallel_size = 1
    parallel_config.prefill_context_parallel_size = 1
    parallel_config.data_parallel_size_local = 1
    parallel_config.numa_bind = False
    parallel_config.assigned_physical_gpu_ids = None
    parallel_config.enable_expert_parallel = False
    vllm_config = MagicMock()
    vllm_config.parallel_config = parallel_config
    vllm_config.kv_transfer_config = None
    vllm_config.ec_transfer_config = None
    return vllm_config


def test_headless_single_rank_follower_runs_worker_inline():
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    vllm_config = _make_follower_vllm_config()

    with (
        patch(
            "vllm.v1.executor.multiproc_executor.set_multiprocessing_worker_envs"
        ) as set_worker_envs,
        patch(
            "vllm.v1.executor.multiproc_executor.set_torch_threads_for_runtime"
        ) as set_runtime_threads,
        patch(
            "vllm.v1.executor.multiproc_executor.WorkerProc.make_worker_process"
        ) as make_worker_process,
        patch(
            "vllm.v1.executor.multiproc_executor.WorkerProc.worker_main"
        ) as worker_main,
        patch(
            "vllm.v1.executor.multiproc_executor.current_platform.is_cpu",
            return_value=False,
        ),
    ):
        executor = MultiprocExecutor(vllm_config, monitor_workers=False)
        try:
            set_worker_envs.assert_called_once_with(1)
            set_runtime_threads.assert_not_called()
            make_worker_process.assert_not_called()
            assert executor.workers == []
            assert executor._inline_worker_kwargs is not None
            assert executor._inline_worker_kwargs["rank"] == 1
            assert executor._inline_worker_kwargs["local_rank"] == 0
            assert executor._inline_worker_kwargs["ready_pipe"] is None
            assert executor._inline_worker_kwargs["death_pipe"] is None

            executor.start_worker_monitor(inline=True)
            worker_main.assert_called_once()
            assert executor._inline_worker_kwargs is None
        finally:
            executor.shutdown()


def test_headless_leader_or_multi_rank_uses_subprocess():
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    vllm_config = _make_follower_vllm_config()
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.monitor_workers = False
    executor.vllm_config = vllm_config
    executor.parallel_config = vllm_config.parallel_config
    executor.local_world_size = 1

    with patch(
        "vllm.v1.executor.multiproc_executor.current_platform.is_cpu",
        return_value=False,
    ):
        assert executor._use_inline_headless_worker() is True
        executor.local_world_size = 2
        assert executor._use_inline_headless_worker() is False
        executor.local_world_size = 1
        executor.parallel_config.node_rank_within_dp = 0
        assert executor._use_inline_headless_worker() is False


def test_inline_worker_main_runs_without_pipes_and_propagates_failure():
    from vllm.v1.executor.multiproc_executor import WorkerProc

    vllm_config = _make_follower_vllm_config()
    fake_worker = MagicMock()
    fake_worker.rank = 1
    fake_worker.rpc_broadcast_mq = MagicMock()
    fake_worker.worker_response_mq = MagicMock()

    with (
        patch(
            "vllm.v1.executor.multiproc_executor.WorkerProc",
            return_value=fake_worker,
        ),
        patch("vllm.v1.executor.multiproc_executor.set_worker_net_device"),
        patch("vllm.v1.executor.multiproc_executor.maybe_init_worker_tracer"),
        patch("vllm.v1.executor.multiproc_executor._trim_host_heap") as trim_heap,
    ):
        WorkerProc.worker_main(
            vllm_config=vllm_config,
            local_rank=0,
            rank=1,
            distributed_init_method="tcp://127.0.0.1:29500",
            input_shm_handle=None,
            ready_pipe=None,
            death_pipe=None,
            shared_worker_lock=MagicMock(),
            is_driver_worker=False,
            inherited_fds=[],
        )
        trim_heap.assert_called_once()
        fake_worker.rpc_broadcast_mq.wait_until_ready.assert_called_once()
        fake_worker.worker_response_mq.wait_until_ready.assert_called_once()
        fake_worker.worker_busy_loop.assert_called_once()
        fake_worker.shutdown.assert_called_once()

        fake_worker.reset_mock()
        fake_worker.worker_busy_loop.side_effect = RuntimeError("rpc loop failed")
        with pytest.raises(RuntimeError, match="rpc loop failed"):
            WorkerProc.worker_main(
                vllm_config=vllm_config,
                local_rank=0,
                rank=1,
                distributed_init_method="tcp://127.0.0.1:29500",
                input_shm_handle=None,
                ready_pipe=None,
                death_pipe=None,
                shared_worker_lock=MagicMock(),
                is_driver_worker=False,
                inherited_fds=[],
            )
        fake_worker.shutdown.assert_called_once()
