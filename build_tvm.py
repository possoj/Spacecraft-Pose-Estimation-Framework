"""
Copyright (c) 2025 Julien Posso
"""

import os
import copy
import torch

from torch.ao.quantization import get_default_qat_qconfig
import torch.quantization.quantize_fx as quantize_fx
from torch.fx import GraphModule

import src.config.train.config as train_cfg
import src.config.build.tvm.config as tvm_cfg
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.modeling.model import import_model, save_model
from src.solver.loss import SPELoss
from src.solver.optimizer import import_optimizer
from src.tools.training import train
from src.tools.utils import select_device, set_seed, prepare_directories
from src.tools.utils import save_score_error
from src.tools.evaluation import evaluation

from src.tvm.rpc_handler import RPCHandler
from src.boards.boards_cfg import import_board
from src.tvm.tvm_compiler import relay_build
from src.spe.spe_torch import SPETorch


def prepare_qat_fx_model(model: torch.nn.Module, qconfig: str) -> GraphModule:
    """
    Prepares a floating-point model for Quantization-Aware Training (QAT) using FX-based quantization.

    Args:
        model (torch.nn.Module): The floating-point model to be prepared for QAT.
        qconfig (str): The quantization configuration, either 'qnnpack' or 'fbgemm'.

    Returns:
        GraphModule: The QAT-prepared model.
    """
    assert qconfig in ('qnnpack', 'fbgemm'), "Invalid qconfig. Choose either 'qnnpack' or 'fbgemm'."
    model.cpu()
    model.train()
    qconfig_dict = {"": get_default_qat_qconfig(qconfig)}

    # Prepare the model for QAT
    qat_model = quantize_fx.prepare_qat_fx(model, qconfig_dict)
    return qat_model


def convert_fx_model(model: GraphModule) -> torch.nn.Module:
    """
    Converts a trained or calibrated QAT/prepared model to its fully quantized int8 version.

    This function replaces floating-point layers with their quantized counterparts,
    optimizing the model for inference.

    References:
        - PyTorch FX Quantization: https://pytorch.org/docs/stable/quantization.html#quantization-aware-training

    Args:
        model (GraphModule): The QAT-trained or prepared model to be converted.

    Returns:
        torch.nn.Module: The fully quantized model.
    """
    model.cpu()
    model.eval()
    int8_model = quantize_fx.convert_fx(model)
    return int8_model


def fuse_model(model: torch.nn.Module) -> GraphModule:
    """
    Applies layer fusion to the model, reducing computation overhead for inference.

    Args:
        model (torch.nn.Module): The model to be fused.

    Returns:
        GraphModule: The fused model optimized for inference.
    """
    model.cpu()
    model.eval()
    fused_model = quantize_fx.fuse_fx(model)
    return fused_model


def main():
    """
    Main function to perform the full pipeline:
    - Load configurations
    - Prepare dataset and model
    - Train with Quantization-Aware Training (QAT)
    - Convert and optimize the model for inference
    - Evaluate the model on different quantization levels
    - Compile the model with TVM for deployment
    """

    # Load TVM configuration: QAT and compilation
    cfg = tvm_cfg.load_config(os.path.join('src','config', 'build', 'tvm', 'config.yaml'))
    pretrained_model_path = cfg.MODEL.PRETRAINED_PATH

    # Load pre-trained model configuration, and reuse most of the same configuration parameters
    cfg_fp32 = train_cfg.load_config(os.path.join(pretrained_model_path, 'config.yaml'))
    camera = load_camera(cfg_fp32.DATA.PATH)
    spe_utils = SPEUtils(
        camera, cfg_fp32.MODEL.HEAD.ORI, cfg_fp32.MODEL.HEAD.N_ORI_BINS_PER_DIM, cfg_fp32.DATA.ORI_SMOOTH_FACTOR,
        cfg_fp32.MODEL.HEAD.ORI_DELETE_UNUSED_BINS, cfg_fp32.MODEL.HEAD.POS, cfg_fp32.MODEL.HEAD.N_POS_BINS_PER_DIM,
        cfg_fp32.DATA.POS_SMOOTH_FACTOR, cfg_fp32.MODEL.HEAD.KEYPOINTS_PATH
    )

    seed = 1001
    set_seed(seed)
    device = select_device()
    data, split = load_dataset(spe_utils, cfg_fp32.DATA.PATH, cfg.DATA.BATCH_SIZE, cfg_fp32.DATA.IMG_SIZE,
                               cfg.DATA.ROT_AUGMENT, cfg.DATA.OTHER_AUGMENT, cfg.DATA.SHUFFLE, seed)

    params_path = os.path.join(pretrained_model_path, 'model', 'parameters.pt')
    bit_width_path = None  # Not a brevitas model

    pytorch_fp32_model, _ = import_model(
        data, cfg_fp32.MODEL.BACKBONE.NAME, cfg_fp32.MODEL.HEAD.NAME, params_path, bit_width_path,
        manual_copy=False, residual=cfg_fp32.MODEL.BACKBONE.RESIDUAL, quantization=cfg_fp32.MODEL.QUANTIZATION,
        ori_mode=cfg_fp32.MODEL.HEAD.ORI, n_ori_bins=spe_utils.orientation.n_bins,
        pos_mode=cfg_fp32.MODEL.HEAD.POS, n_pos_bins=spe_utils.position.n_bins,
    )
    spe_model = SPETorch(pytorch_fp32_model, torch.device("cpu"), spe_utils)

    # Prepare model for QAT. Use qnnpack for ARM inference
    pytorch_qat_model = prepare_qat_fx_model(copy.deepcopy(pytorch_fp32_model), 'qnnpack')

    # Quantization Aware Training (QAT)
    if cfg.TRAIN.QAT:
        print(f"\n\n----- Quantization Aware Training -----\n")

        save_folder = prepare_directories(
            'experiments', 'train',
            f'{os.path.basename(pretrained_model_path)}'.replace('fp32', 'tvm_qat'),
            ('model', 'results', 'tensorboard')
        )
        tvm_cfg.save_config(cfg, os.path.join(save_folder, 'config.yaml'))

        tensorboard_cfg = {
            'log_folder': os.path.join(save_folder, 'tensorboard'),
            'save_model': True,
            'save_parameters': False,
        }

        spe_loss = SPELoss(cfg_fp32.MODEL.HEAD.ORI, cfg_fp32.MODEL.HEAD.POS, beta=1, norm_distance=True)
        # Optimizer uses TVM build configuration file
        optimizer, scheduler = import_optimizer(
            pytorch_qat_model, cfg.TRAIN.LR, cfg.TRAIN.OPTIM, cfg.TRAIN.MOMENTUM, cfg.TRAIN.DECAY,
            cfg.TRAIN.SCHEDULER, cfg.TRAIN.MILESTONES, cfg.TRAIN.GAMMA, verbose=True
        )

        # Training
        pytorch_qat_model, loss, score, error = train(
            pytorch_qat_model, data, cfg.TRAIN.N_EPOCH, spe_utils, spe_loss, scheduler,
            optimizer, tensorboard_cfg, split['train'], device, cfg.TRAIN.CLIP_BATCHNORM, amp=False
        )
        save_score_error(score, error, path=os.path.join(save_folder, 'results'), name='train.xlsx')

        # Evaluation
        spe_model.update_model(pytorch_qat_model, device)
        score, error = evaluation(spe_model, data, spe_utils, split['eval'])
        spe_model.delete_model()
        save_score_error(score, error, path=os.path.join(save_folder, 'results'), name='eval.xlsx')

        # Save model
        save_model(os.path.join(save_folder, 'model'), pytorch_qat_model)

    else:
        # if not QAT, load the QAT model
        qat_params_path = os.path.join(
            'experiments', 'train',
            f'{os.path.basename(pretrained_model_path)}'.replace('fp32', 'tvm_qat'),
            'model', 'parameters.pt'
        )
        assert(os.path.isfile(qat_params_path)), (f'{qat_params_path} not found, you need to perform Quantization Aware '
                                                  f'Training.')
        pytorch_qat_model.load_state_dict(torch.load(qat_params_path))

    # Prepare model for inference: finalize quantization and fuse layers
    pytorch_int8_model = convert_fx_model(pytorch_qat_model)
    pytorch_int8_fused_model = fuse_model(pytorch_int8_model)

    # Load data for evaluation and TorchScript compilation
    data, split = load_dataset(spe_utils, cfg_fp32.DATA.PATH, 1, cfg_fp32.DATA.IMG_SIZE,
                               False, False, False)

    # Build folder to save TVM models
    save_folder = prepare_directories(
        'experiments', 'build',
        f'tvm/{os.path.basename(pretrained_model_path)}'.replace('fp32', 'tvm_qat'),
        ('model', 'eval_host', 'on_board')
    )
    tvm_cfg.save_config(cfg, os.path.join(save_folder, 'config.yaml'))
    # Also save fp32 cfg as it contains interesting information
    train_cfg.save_config(cfg_fp32, os.path.join(save_folder, 'config_fp32.yaml'))

    # Get example image
    img, _ = next(iter(data[split['eval'][0]]))
    img = img['torch']

    with torch.no_grad():
        pytorch_int8_fused_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()  # Free any lingering memory
        pytorch_int8_fused_model.eval()
        ts_int8_model = torch.jit.trace(pytorch_int8_fused_model, img.cpu())
        torch.jit.save(
            ts_int8_model,
            os.path.join(save_folder, 'model', 'jit_model.pt')
        )
    ts_int8_model = torch.jit.load(os.path.join(save_folder, 'model', 'jit_model.pt')).eval().cpu()

    # EVAL ON HOST
    models = {
        'fp32': pytorch_fp32_model,
        'qat': pytorch_qat_model,
        'int8': pytorch_int8_model,
        'int8_fused': pytorch_int8_fused_model,
        'ts_int8_fused': ts_int8_model
    }
    for desc, model in models.items():
        print(f"\nEval {desc} model")
        spe_model.update_model(model, torch.device("cpu"))
        score, error = evaluation(spe_model, data, spe_utils, split['eval'])
        spe_model.delete_model()
        save_score_error(score, error, path=os.path.join(save_folder, 'eval_host'), name=f'eval_{desc}.xlsx')

    # TVM toolchain
    board = import_board(cfg.COMPILE.BOARD)

    # Start RPC server on the host and register target board to the RPC server
    # Only if we need to run TVM auto-scheduling
    start_rpc = cfg.COMPILE.RUN_AUTOSCHEDULING
    rpc_handler = RPCHandler(board, start_tracker=start_rpc, start_ssh=start_rpc, print_ssh=start_rpc)
    host_ip = rpc_handler.get_host_ip()

    print("TVM compilation")
    input_name = "input0"
    tvm_model = relay_build(
        ts_int8_model, tuple(img.shape), input_name, board, host_ip,
        cfg.COMPILE.RUN_AUTOSCHEDULING, cfg.COMPILE.N_TRIALS, cfg.COMPILE.SCHEDULING_PATH
    )

    # Save the compiled library
    print(f"Save compiled TVM model to {os.path.join(save_folder, 'model', 'lib.tar')}")
    tvm_model['lib'].export_library(os.path.join(save_folder, 'model', 'lib.tar'))

    if start_rpc:
        # Close SSH session
        rpc_handler.close_ssh_thread()


if __name__ == "__main__":
    main()
