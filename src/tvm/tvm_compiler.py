"""
Copyright (c) 2025 Julien Posso
"""

import os
import tvm
import torch
from tvm import relay, auto_scheduler
from typing import Any, Tuple, Dict, List


def extract_graph(
    ts_model: torch.jit.ScriptModule,
    input_shape: Tuple[int, ...],
    input_name: str
) -> Tuple[tvm.IRModule, Dict]:
    """
    Load a TorchScript graph and convert it to a Relay graph.

    Args:
        ts_model (torch.jit.ScriptModule): The TorchScript model to be converted.
        input_shape (Tuple[int, ...]): The shape of the model input.
        input_name (str): The name of the input tensor.

    Returns:
        Tuple[tvm.IRModule, Dict[str, Any]]: A tuple containing the Relay graph (IRModule) and its parameters.
    """
    shape_list = [(input_name, input_shape)]
    relay_graph, params = relay.frontend.from_pytorch(ts_model, shape_list)
    return relay_graph, params


def run_tuning(
    num_trials: int,
    tasks: List[Any],
    task_weights: List[float],
    device_key: str,
    rpc_host: str,
    rpc_port: int,
    log_file: str
) -> None:
    """
    Run auto-scheduling to optimize the inference scheduling on the target device.

    Args:
        num_trials (int): The number of measurement trials for tuning.
        tasks (List[Any]): A list of tasks extracted from the Relay graph.
        task_weights (List[float]): Weights corresponding to the importance of each task.
        device_key (str): Device key (usually the lower-cased board name).
        rpc_host (str): The RPC host IP address.
        rpc_port (int): The RPC port number.
        log_file (str): Path to the file where tuning logs will be recorded.

    Returns:
        None
    """
    print("Begin auto-scheduling...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,
        runner=auto_scheduler.RPCRunner(
            device_key,
            host=rpc_host,
            port=rpc_port,
            timeout=30,
            repeat=1,
            min_repeat_ms=200,
            enable_cpu_cache_flush=True,
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)
    print('Auto scheduling done')


def relay_build(
    ts_model: torch.jit.ScriptModule,
    input_shape: Tuple[int, ...],
    input_name: str, board: Any,
    rpc_host: str,
    auto_scheduling: bool = False,
    num_trials: int = 100,
    logfile: str = 'tuning2.json'
) -> Dict[str, Any]:
    """
    Compile a TorchScript model to machine code with optional auto-scheduling.

    Args:
        ts_model (torch.jit.ScriptModule): The TorchScript model to compile.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        input_name (str): The name of the input tensor.
        board (Any): An object representing the target board; it must have attributes 'name' and 'port'.
        rpc_host (str): The RPC host address.
        auto_scheduling (bool, optional): Whether to perform auto-scheduling. Defaults to False.
        num_trials (int, optional): Number of tuning trials if auto-scheduling is enabled. Defaults to 100.
        logfile (str, optional): Path to the tuning log file. Defaults to 'tuning2.json'.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "lib": The compiled library module.
            - "relay_graph": The Relay graph (IRModule).
            - "params": The parameters dictionary.
    """
    target = tvm.target.Target('llvm -mtriple=aarch64-linux-gnu -mattr=+neon')  # ARM target
    device_key = board.name.lower()
    rpc_port = board.port

    relay_graph, params = extract_graph(ts_model, input_shape, input_name)

    if auto_scheduling:
        tasks, task_weights = auto_scheduler.extract_tasks(relay_graph["main"], params, target)
        print(f"Auto-scheduling should run at least for {len(tasks) * 800} trials")
        run_tuning(num_trials, tasks, task_weights, device_key, rpc_host, rpc_port, logfile)
    else:
        # Ensure the tuning history file exists, even if auto-scheduling is disabled.
        assert os.path.exists(logfile), f"Tuning log file {logfile} does not exist."

    with auto_scheduler.ApplyHistoryBest(logfile):
        with tvm.transform.PassContext(opt_level=4, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(relay_graph, target=target, params=params)

    return {
        "lib": lib,
        "relay_graph": relay_graph,
        "params": params
    }
