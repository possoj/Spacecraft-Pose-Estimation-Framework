"""
Copyright (c) 2025 Julien Posso
"""

import os
import torch
import torch_tensorrt
from tqdm import tqdm
from typing import List, Dict

# PyTorch Quantization Toolkit
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.calib import MaxCalibrator

# Configurations & Utilities
import src.config.train.config as train_cfg
import src.config.build.nvidia.config as nvidia_cfg
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.modeling.model import import_model, save_model
from src.solver.loss import SPELoss
from src.solver.optimizer import import_optimizer
from src.tools.training import train
from src.tools.utils import select_device, set_seed, prepare_directories, save_score_error
from src.tools.evaluation import evaluation
from src.spe.spe_torch import SPETorch


def compute_amax(model: torch.nn.Module, method: str = "max") -> None:
    """
    Applies calibration to the quantized tensors in the model.

    Args:
        model (torch.nn.Module): PyTorch model with pytorch_quantization.
        method (str): Calibration method ("max", "histogram", "mse", "entropy").
    """
    print(f"Applying calibration with method: {method.upper()}")

    model.cuda()
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
            if isinstance(module._calibrator, MaxCalibrator):
                module.load_calib_amax()
            else:
                module.load_calib_amax(method=method)
            print(f"{name:40}: Calibrated with {method.upper()}")

    model.cuda()  # Ensure the model is on GPU after calibration


def collect_stats(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, num_batches: int) -> None:
    """
    Collects statistics on activations by temporarily disabling quantization.

    Args:
        model (torch.nn.Module): PyTorch model.
        data_loader (torch.utils.data.DataLoader): DataLoader with calibration images.
        num_batches (int): Number of batches to use for calibration.
    """
    print("Collecting calibration statistics...")

    model.cuda()

    # Enable calibration mode
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
            module.disable_quant()
            module.enable_calib()

    # Feed data to the network to collect statistics
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image["torch"].cuda())
        if i >= num_batches:
            break

    # Disable calibration and enable quantization
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
            module.enable_quant()
            module.disable_calib()

    print("Calibration statistics collected.")


def calibrate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        num_calib_batch: int,
        calib_type: str = "max",
        hist_percentile: List[float] = [99.99],
) -> torch.nn.Module:
    """
    Performs model calibration using the selected method.

    Args:
        model (torch.nn.Module): PyTorch quantized model.
        data_loader (torch.utils.data.DataLoader): Calibration dataset DataLoader.
        num_calib_batch (int): Number of batches to use for calibration.
        calib_type (str): Calibration type ("max", "histogram", "mse", "entropy").
        hist_percentile (List[float]): Percentiles to use for histogram calibration.
    Returns:
        torch.nn.Module: The calibrated model.
    """
    if num_calib_batch > 0:
        print(f"Starting calibration with method: {calib_type.upper()}")

        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch)

        if calib_type == "max":
            compute_amax(model, method="max")

        elif calib_type == "histogram":
            for percentile in hist_percentile:
                print(f"Histogram calibration with {percentile}%")
                compute_amax(model, method="percentile")

        elif calib_type in ["mse", "entropy"]:
            print(f"Advanced calibration with {calib_type.upper()}")
            compute_amax(model, method=calib_type)

        else:
            raise ValueError(
                f"Unknown calibration method '{calib_type}'! Choose from ['max', 'histogram', 'mse', 'entropy']")

    return model


def main():
    """
    Main function to perform:
    - Model loading and dataset preparation
    - Model quantization and calibration
    - Training with Quantization-Aware Training (QAT)
    - Evaluation and TensorRT compilation
    """

    # Load Nvidia TensorRT configuration: QAT and compilation
    cfg = nvidia_cfg.load_config(os.path.join('src','config', 'build', 'nvidia', 'config.yaml'))
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

    # Load FP32 model
    params_path = os.path.join(pretrained_model_path, 'model', 'parameters.pt')
    bit_width_path = None  # Not a brevitas model
    pytorch_fp32_model, _ = import_model(
        data, cfg_fp32.MODEL.BACKBONE.NAME, cfg_fp32.MODEL.HEAD.NAME, params_path, bit_width_path,
        manual_copy=False, residual=cfg_fp32.MODEL.BACKBONE.RESIDUAL, quantization=cfg_fp32.MODEL.QUANTIZATION,
        ori_mode=cfg_fp32.MODEL.HEAD.ORI, n_ori_bins=spe_utils.orientation.n_bins,
        pos_mode=cfg_fp32.MODEL.HEAD.POS, n_pos_bins=spe_utils.position.n_bins,
    )
    spe_model = SPETorch(pytorch_fp32_model, torch.device("cpu"), spe_utils)

    # Import the model. All the regular conv and FC layers will be converted to their quantized counterparts due
    # to quant_modules.initialize()
    quant_modules.initialize()
    pytorch_qat_model, _ = import_model(
        data, cfg_fp32.MODEL.BACKBONE.NAME, cfg_fp32.MODEL.HEAD.NAME, params_path, bit_width_path,
        manual_copy=False, residual=cfg_fp32.MODEL.BACKBONE.RESIDUAL, quantization=cfg_fp32.MODEL.QUANTIZATION,
        ori_mode=cfg_fp32.MODEL.HEAD.ORI, n_ori_bins=spe_utils.orientation.n_bins,
        pos_mode=cfg_fp32.MODEL.HEAD.POS, n_pos_bins=spe_utils.position.n_bins,
    )
    pytorch_qat_model = pytorch_qat_model.to(device)

    # Quantization Aware Training (QAT)
    if cfg.TRAIN.QAT:
        print(f"\n\n----- Quantization Aware Training -----\n")

        save_folder = prepare_directories(
            'experiments', 'train',
            f'{os.path.basename(pretrained_model_path)}'.replace('fp32', 'nvidia_qat'),
            ('model', 'results', 'tensorboard')
        )
        nvidia_cfg.save_config(cfg, os.path.join(save_folder, 'config.yaml'))
        train_cfg.save_config(cfg_fp32, os.path.join(save_folder, 'config_fp32.yaml'))

        tensorboard_cfg = {
            'log_folder': os.path.join(save_folder, 'tensorboard'),
            'save_model': True,
            'save_parameters': False,
        }

        spe_loss = SPELoss(cfg_fp32.MODEL.HEAD.ORI, cfg_fp32.MODEL.HEAD.POS, beta=1, norm_distance=True)
        # Optimizer uses nvidia build configuration file
        optimizer, scheduler = import_optimizer(
            pytorch_qat_model, cfg.TRAIN.LR, cfg.TRAIN.OPTIM, cfg.TRAIN.MOMENTUM, cfg.TRAIN.DECAY,
            cfg.TRAIN.SCHEDULER, cfg.TRAIN.MILESTONES, cfg.TRAIN.GAMMA, verbose=True
        )

        # Calibrate the model using percentile calibration technique.
        # Initializes  scaling factors and zero points based on activation statistics, reducing the impact of outliers
        # and providing a more stable starting point for the training process to adapt effectively to low-precision
        # representation.
        with torch.no_grad():
            pytorch_qat_model = calibrate_model(
                model=pytorch_qat_model,
                data_loader=data['train'],
                num_calib_batch=256,
                calib_type="histogram",
                # hist_percentile=[99.9, 99.99, 99.999, 99.9999],
                hist_percentile=[99.99],
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
            f'{os.path.basename(pretrained_model_path)}'.replace('fp32', 'nvidia_qat'),
            'model', 'parameters.pt'
        )
        assert(os.path.isfile(qat_params_path)), (f'{qat_params_path} not found, you need to perform Quantization Aware '
                                                  f'Training.')
        pytorch_qat_model.load_state_dict(torch.load(qat_params_path))

        # It seems the amax values to compute the scaling factors are not properly saved and loaded when
        # saving/loading model parameters (scaling factors are not directly stored in the model state dict: TensorFlow
        # convention). Evaluate the model before compiling it with TensorRT fix the issue. Else, the accuracy drops
        spe_model.update_model(pytorch_qat_model, device)
        _ = evaluation(spe_model, data, spe_utils, split['eval'])
        spe_model.delete_model()


    # Nvidia TensorRT Compilation and evaluation on host
    print(f"\n\n----- Nvidia TensorRT compilation -----\n")
    assert device.type == "cuda", "Need a cuda GPU on the host to compile"

    # Load data for evaluation and TensorRT compilation
    data, split = load_dataset(spe_utils, cfg_fp32.DATA.PATH, 1, cfg_fp32.DATA.IMG_SIZE,
                               False, False, False)

    # Get example image
    img, _ = next(iter(data[split['eval'][0]]))
    img = img['torch']

    # Build folder to save Nvidia TensorRT models
    save_folder = prepare_directories(
        'experiments', 'build',
        f'nvidia/{os.path.basename(pretrained_model_path)}'.replace('fp32', 'nvidia_qat'),
        ('model', 'eval_host', 'on_board')
    )
    nvidia_cfg.save_config(cfg, os.path.join(save_folder, 'config.yaml'))
    train_cfg.save_config(cfg_fp32, os.path.join(save_folder, 'config_fp32.yaml'))

    # enables the QAT model to use torch.fake_quantize_per_tensor_affine and torch.fake_quantize_per_channel_affine
    # operators instead of tensor_quant function to export quantization operators
    # Very slightly drop in evaluation metrics
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # TorchScript JIT trace and save model.
    # Very slightly drop in evaluation metrics
    with torch.no_grad():
        pytorch_qat_model.to(device)
        jit_model = torch.jit.trace(pytorch_qat_model, img.to(device))
        torch.jit.save(
            jit_model,
            os.path.join(save_folder, 'model', 'jit_model.pt')
        )
    # Loading the TorchScript model and compiling it into a TensorRT model:
    # Follow Nvidia guidelines: save and then load back the model
    jit_model = torch.jit.load(os.path.join(save_folder, 'model', 'jit_model.pt')).eval().to(device)

    # Export the model to ONNX for visualization
    # We clearly see the quant/dequant nodes
    dummy_input = torch.randn(list(img.size())).to(device)
    torch.onnx.export(
        jit_model,  # TorchScript model
        dummy_input,  # Example input
        os.path.join(save_folder, "model", "jit_model.onnx"),  # Output ONNX file
        verbose=False,  # Include detailed model information
        input_names=["input"],  # Name of input nodes
        output_names=["output"],  # Name of output nodes
        opset_version=13  # ONNX opset version
    )

    # Check if there are parameters outside the int8 range.
    # move it before tensorRT compilation
    # print('Parameters check')
    # check_parameters_int8_range(qat_model)

    compile_spec = {
        "inputs": [torch_tensorrt.Input(list(img.size()))],
        "enabled_precisions": torch.int8,
        # "enabled_precisions": {torch.int8, torch.float32},  # Allow fallback to FP32 if needed
        "ir": "ts",
        # "logging_level": torch_tensorrt.logging.Level.Debug  # Enable verbose logging,
    }
    # Set the global logging level to DEBUG
    # trt_logging.set_reportable_log_level(trt_logging.Level.Debug)
    # trt_logging.set_reportable_log_level(trt_logging.Level.Warning)
    # trt_logging.set_reportable_log_level(trt_logging.Level.Graph)

    # Important: torch_tensorrt.compile(...) does not produce a standalone TensorRT engine (.engine),
    # but rather a TorchScript model with embedded TensorRT-optimized subgraphs.
    model_int8_trt = torch_tensorrt.compile(jit_model, **compile_spec)
    torch.jit.save(
        model_int8_trt,
        os.path.join(save_folder, 'model', 'model_int8_trt_host.ts')
    )

    # Eval on host
    models = {
        'fp32': pytorch_fp32_model,
        'int8_pytorch': pytorch_qat_model,
        'int8_jit': jit_model,
        'model_int8_trt': model_int8_trt
    }
    for desc, model in models.items():
        print(f"\nEval {desc} model")
        spe_model.update_model(model, device)
        score, error = evaluation(spe_model, data, spe_utils, split['eval'])
        spe_model.delete_model()
        save_score_error(score, error, path=os.path.join(save_folder, 'eval_host'), name=f'eval_{desc}.xlsx')


if __name__ == "__main__":
    main()
