"""
Copyright (c) 2025 Julien Posso
"""

import sys
from typing import List, Dict, Tuple, Union, Any
from tqdm import tqdm
import torch
import numpy as np

from src.spe.spe_utils import SPEUtils
from src.tools.utils import RunningAverage
from src.spe.spe_torch import SPETorch


def mad(data: list):
    """
    Compute median absolute deviation of the data

    Args:
        data (list): a list of float values

    Returns:
        float: median absolute deviation of input data

    """
    # Compute the median of the data
    median = np.median(data)
    # Compute the absolute deviations from the median
    absolute_deviations = np.abs(np.array(data) - median)
    # Compute the median of these absolute deviations (MAD)
    return np.median(absolute_deviations).tolist()


def evaluation(
    spe_model: Any,
    dataloader: Dict[str, torch.utils.data.DataLoader],
    spe_utils: SPEUtils,
    split: Tuple[str, ...] = ('test', 'valid'),
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    """
    Evaluate the model on specified splits and record scores, and errors.

    Args:
        spe_model (Union[SPETorch, SPETVMARM]): The Spacecraft Pose Estimation model (wrapper class) to evaluate.
        dataloader (Dict[str, torch.utils.data.DataLoader]): A dictionary containing the data loaders for different splits.
        spe_utils (SPEUtils): An instance of the SPEUtils class.
        split (Tuple[str, ...], optional): A tuple containing the split names (train, valid, ...). Defaults to ('test', 'valid').

    Returns:
        Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]: A tuple containing the recorded
        scores and errors.
    """

    # Record loss/score/error during evaluation
    rec_score = {x: {'ori': [], 'pos': [], 'esa': []} for x in split}
    rec_error = {x: {'ori': [], 'pos': [], 'ori_std': [], 'pos_std': [], 'ori_mad': [], 'pos_mad': []} for x in split}

    for phase in split:
        # Store ori and pos errors for later computation of standard deviation
        error = {'ori': [], 'pos': []}

        running_avg = RunningAverage(keys=('esa_score', 'ori_score', 'pos_score', 'ori_error', 'pos_error'))

        # Batch loop
        loop = tqdm(dataloader[phase], desc=f"Eval - {phase}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    ncols=115, file=sys.stdout)

        for i, (images, targets) in enumerate(loop):
            # Inference
            pose, _ = spe_model.predict(images['torch'])

            # Compute evaluation metrics
            targets = {key: value.detach().cpu().numpy() for key, value in targets.items()}
            eval_metrics = spe_utils.get_score(targets, pose)
            running_avg.update(eval_metrics, images['torch'].size(0))

            # Update progress bar
            loop.set_postfix(running_avg.get_multiple(keys=('ori_error', 'pos_error', 'esa_score')))

            # Store errors for STD computation
            error['pos'].extend(np.linalg.norm(targets['pos'] - pose['pos'], axis=1))
            inter_sum = np.abs(np.sum(pose['ori'] * targets['ori'] , axis=1, keepdims=True))
            inter_sum[inter_sum > 1] = 1
            error['ori'].extend((2 * np.arccos(inter_sum) * 180 / np.pi).reshape(-1))

        # Store scores and errors for printing
        rec_score[phase]['ori'].append(running_avg.get('ori_score'))
        rec_score[phase]['pos'].append(running_avg.get('pos_score'))
        rec_score[phase]['esa'].append(running_avg.get('esa_score'))
        rec_error[phase]['ori'].append(running_avg.get('ori_error'))
        rec_error[phase]['pos'].append(running_avg.get('pos_error'))

        # Store STD
        rec_error[phase]['ori_std'].append(np.std(error['ori']).tolist())
        rec_error[phase]['pos_std'].append(np.std(error['pos']).tolist())
        # Also store Median Absolute Deviation (more robust to outliers than STD)
        rec_error[phase]['ori_mad'].append(mad(error['ori']))
        rec_error[phase]['pos_mad'].append(mad(error['pos']))


    return rec_score, rec_error
