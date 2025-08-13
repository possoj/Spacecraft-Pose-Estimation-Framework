import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

import os

from src.config.train.config import load_config
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.modeling.model import import_model
from src.tools.utils import select_device, set_seed


# Custom summary function to display detailed layer stats
def detailed_model_summary(model, input_size, include_batchnorm=True):
    layer_stats = []
    summary_by_type = defaultdict(lambda: {"MACs": 0, "Params": 0})
    total_macs = 0
    total_params = 0

    # Hook function to capture input/output shapes
    def hook_fn(layer, layer_input, layer_output):
        input_shape = layer_input[0].size()
        output_shape = layer_output.size()

        layer_type = layer.__class__.__name__
        num_params = sum(p.numel() for p in layer.parameters())

        # Calculate MACs based on the layer type
        if isinstance(layer, nn.Conv2d):
            output_elements = np.prod(output_shape[1:])  # Exclude batch size
            kernel_elements = np.prod(layer.kernel_size)  # kernel width * height
            mac_operations = kernel_elements * layer.in_channels * output_elements // layer.groups
            extra_info = f"Kernel: {layer.kernel_size}; Stride: {layer.stride}; Padding: {layer.padding}"
        elif isinstance(layer, nn.Linear):
            mac_operations = input_shape[1] * output_shape[1]
            extra_info = f"In Feat: {layer.in_features}; Out Feat: {layer.out_features}"
        elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mac_operations = np.prod(output_shape) * 2  # For mean/variance calculations
            extra_info = f"Num Features: {layer.num_features}"
        else:
            mac_operations = 0
            extra_info = ""

        # Store layer info
        layer_info = {
            "Layer Type": layer_type,
            "Input Shape": list(input_shape),
            "Output Shape": list(output_shape),
            "Params": num_params,
            "MACs": mac_operations,
            "Extra Info": extra_info,
        }
        layer_stats.append(layer_info)

        # Update the summary by type
        summary_by_type[layer_type]["MACs"] += mac_operations
        summary_by_type[layer_type]["Params"] += num_params

        # Update total MACs and params
        nonlocal total_macs, total_params
        total_macs += mac_operations
        total_params += num_params

    # Register hooks for all layers
    hooks = []
    for layer in model.modules():
        if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and len(
                list(layer.children())) == 0:

            if include_batchnorm or not isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                hooks.append(layer.register_forward_hook(hook_fn))

    # Forward pass to capture stats
    x = torch.randn(input_size)
    model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print per-layer stats
    print(f"{'Layer Type':<17} {'Input Shape':<20} {'Output Shape':<20} {'Params':<15} {'MACs':<15} {'Extra Info':<50}")
    print("=" * 150)
    for layer in layer_stats:
        params_str = f"{layer['Params']:,}"  # Format with commas
        macs_str = f"{layer['MACs']:,}"  # Format with commas
        print(
            f"{layer['Layer Type']:<17} {str(layer['Input Shape']):<20} {str(layer['Output Shape']):<20} {params_str:<15} {macs_str:<15} {layer['Extra Info']:<50}")

    # Print per-layer type MAC and Params summary
    print("\nSummary by Layer Type:")
    print(f"{'Layer Type':<20} {'Total Params':<20} {'Total MACs':<20}")
    print("=" * 60)
    for layer_type, stats in summary_by_type.items():
        params_str = f"{stats['Params']:,}"  # Format with commas
        macs_str = f"{stats['MACs']:,}"  # Format with commas
        print(f"{layer_type:<20} {params_str:<20} {macs_str:<20}")

    # Print total MACs and Params
    print("\nTotal MACs and Parameters:")
    print(f"{'Total Params':<20}: {total_params:,}")
    print(f"{'Total MACs':<20}: {total_macs:,}")


def main():
    seed = 1001
    set_seed(seed)
    device = select_device()

    exp_dir = input('select path to experiment/model to evaluate: ')

    params_path = os.path.join(exp_dir, 'model', 'parameters.pt')
    bw_path = os.path.join(exp_dir, 'model', 'bit_width.json')
    bit_width_path = bw_path if os.path.exists(bw_path) else None

    cfg = load_config(os.path.join(exp_dir, 'config.yaml'))

    camera = load_camera(cfg.DATA.PATH)
    spe_utils = SPEUtils(
        camera, cfg.MODEL.HEAD.ORI, cfg.MODEL.HEAD.N_ORI_BINS_PER_DIM, cfg.DATA.ORI_SMOOTH_FACTOR,
        cfg.MODEL.HEAD.ORI_DELETE_UNUSED_BINS, cfg.MODEL.HEAD.POS, cfg.MODEL.HEAD.N_POS_BINS_PER_DIM,
        cfg.DATA.POS_SMOOTH_FACTOR, cfg.MODEL.HEAD.KEYPOINTS_PATH
    )

    # batch_size = cfg.DATA.BATCH_SIZE
    batch_size = 1
    rot_augment = False
    other_augment = False
    shuffle = False
    data, split = load_dataset(spe_utils, cfg.DATA.PATH, batch_size, cfg.DATA.IMG_SIZE,
                               rot_augment, other_augment, shuffle, seed)

    manual_copy = False
    model, _ = import_model(
        data, cfg.MODEL.BACKBONE.NAME, cfg.MODEL.HEAD.NAME, params_path, bit_width_path,
        manual_copy, residual=cfg.MODEL.BACKBONE.RESIDUAL, quantization=cfg.MODEL.QUANTIZATION,
        ori_mode=cfg.MODEL.HEAD.ORI, n_ori_bins=spe_utils.orientation.n_bins,
        pos_mode=cfg.MODEL.HEAD.POS, n_pos_bins=spe_utils.position.n_bins,
    )

    model.eval()

    img, _ = next(iter(data['train']))
    shape = list(img['torch'].shape)

    detailed_model_summary(model, shape, include_batchnorm=False)


if __name__ == '__main__':
    main()
