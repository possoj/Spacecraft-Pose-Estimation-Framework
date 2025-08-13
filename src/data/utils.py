"""
Copyright (c) 2024 Julien Posso
"""

import random
import copy
import json
import os
import re
import cv2

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.spe.utils import euler2dcm, dcm2quat, multiply_quaternions


class BrightnessContrast(object):
    """ Adjust brightness and contrast of the image in a fashion of
        OpenCV's convertScaleAbs, where

        newImage = alpha * image + beta

        image: torch.Tensor image (0 ~ 1)
        alpha: multiplicative factor
        beta:  additive factor (0 ~ 255)
    """
    def __init__(self, alpha=(0.5, 2.0), beta=(-25, 25)):
        self.alpha = torch.tensor(alpha).log()
        self.beta = torch.tensor(beta)/255

    def __call__(self, image):
        # Contrast - multiplicative factor
        loga = torch.rand(1) * (self.alpha[1] - self.alpha[0]) + self.alpha[0]
        a = loga.exp()

        # Brightness - additive factor
        b = torch.rand(1) * (self.beta[1] - self.beta[0]) + self.beta[0]

        # Apply
        image = torch.clamp(a*image + b, 0, 1)

        return image


class GaussianNoise(object):
    """ Add random Gaussian white noise

        image: torch.Tensor image (0 ~ 1)
        std:   noise standard deviation (0 ~ 255)
    """
    def __init__(self, std=25):
        self.std = std/255

    def __call__(self, image):
        noise = torch.randn(image.shape, dtype=torch.float32) * self.std
        image = torch.clamp(image + noise, 0, 1)
        return image


class AddGaussianNoise(object):
    """
    Add Gaussian noise to a tensor.

    Args:
        mean (float): Mean of the Gaussian distribution.
        std (float): Standard deviation of the Gaussian distribution.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """Apply Gaussian noise to a tensor"""
        return torch.abs(tensor + torch.randn(tensor.size()) * self.std + self.mean)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomRotation:
    """
    A custom transformation class for rotating images and corresponding pose data in spacecraft pose estimation tasks.
    This transformation randomly applies rotation to the input image and updates the orientation and position data accordingly.
    """

    def __init__(self, spe_utils, rot_probability: float = 0.5, rot_max_magnitude: float = 50.0):
        """
        Initializes the custom rotation transformation.
        Args:
            spe_utils: Instance used for performing image and pose transformations.
            rot_probability: Probability of applying a rotation. Defaults to 0.5.
            rot_max_magnitude: Maximum rotation magnitude in degrees. Defaults to 50.0.
        """
        self.spe_utils = spe_utils
        self.rot_probability = rot_probability
        self.rot_max_magnitude = rot_max_magnitude

    def __call__(self, image: Image.Image, pose: dict) -> (Image.Image, dict):
        """
        Apply the custom transformation to an image and its corresponding pose data. This method randomly decides
        whether to apply a rotation based on the instance's rotation probability. If a rotation is applied, it rotates
        the image and updates the pose data accordingly.

        Args:
            image (Image.Image): The input image to be transformed. Expected to be a PIL Image object.
            pose (dict): The pose data associated with the image. This dictionary must contain two keys: 'ori' and 'pos',
                         each holding a torch.Tensor. 'ori' represents the orientation, and 'pos' represents the position
                         of the spacecraft.

        Returns:
            tuple: A tuple containing two elements:
                - A PIL.Image.Image representing the transformed image.
                - A dictionary with the updated pose data. The dictionary will have the same structure as the input `pose`
                  but with updated values for 'ori' and 'pos', reflecting the changes made by the applied rotation.
        """
        ori = pose['ori']
        pos = pose['pos']

        # Apply random rotations to some images only
        if np.random.rand() < self.rot_probability:
            image_np = np.array(image)
            # Randomly choose the rotation magnitude
            rotation_deg = (np.random.rand() - 0.5) * 2 * self.rot_max_magnitude
            # Create a rotation matrix on yaw axis
            r_change = euler2dcm(rotation_deg, 0, 0)
            # Apply a rotation around yaw axis on the image frame, taking into account the camera parameters
            transformation_matrix = np.dot(np.dot(self.spe_utils.camera.K, r_change),
                                           np.linalg.inv(self.spe_utils.camera.K))
            # Get image dimensions
            h, w = image_np.shape[:2]

            # Rotate image
            image_warped = cv2.warpPerspective(image_np, transformation_matrix, (w, h))

            # Update pose
            pos_new = torch.tensor(np.dot(pos, r_change.T), dtype=torch.float32)
            ori_new = torch.tensor(multiply_quaternions(dcm2quat(r_change), ori.numpy()), dtype=torch.float32)

            image = Image.fromarray(image_warped)
            pose = {'ori': ori_new, 'pos': pos_new}

        return image, pose


def get_image_number(path_and_pose):
    image_name = os.path.basename(path_and_pose[0])
    numbers_only = re.sub(r'[^0-9]', '', image_name)
    return int(numbers_only)


def find_key_in_dict(dict_to_find, keys_list):
    """Finds and returns the first key from keys_list that exists in the dictionary.

    Args:
        dict_to_find (dict): The dictionary to search.
        keys_list (list): A list of keys to check in the dictionary.

    Returns:
        The key from keys_list that exists in the dictionary, or None if none exist.
    """
    for key in keys_list:
        if key in dict_to_find:
            return key
    return None  # If no key is found in the dictionary


class SPEDataset(Dataset):
    """
    A dataset class for spacecraft pose estimation tasks, capable of applying transformations to images and their corresponding pose labels.

    The class handles loading and transforming images and labels for training and evaluation in spacecraft pose estimation models.

    Args:
        spe_utils (SPEUtils): The SPEUtils instance used for encoding orientation and other utility operations.
        transform (callable, optional): Standard PyTorch transform to be applied to each image.
        rot_transform (CustomTransform, optional): Custom rotation transform that acts on both images and their corresponding pose data.
        images_path (str, optional): The directory path where the images are stored. Default is "../datasets/images".
        labels_path (str, optional): The file path of the JSON file containing the labels. Default is "../datasets/labels.json".
    """

    def __init__(self, spe_utils, transform=None, rot_transform=None, images_path="../datasets/images",
                 labels_path="../datasets/labels.json"):
        self.spe_utils = spe_utils
        self.transform = transform
        self.rot_transform = rot_transform

        # Load labels from JSON file
        with open(labels_path, 'r') as f:
            target_list = json.load(f)

        # Organize data into a dictionary with image paths, orientations, and positions
        ori_name = find_key_in_dict(target_list[0], ['q', 'q_vbs2tango', 'q_vbs2tango_true'])
        pos_name = find_key_in_dict(target_list[0], ['t', 'r_Vo2To_vbs_true'])
        self.data = {
            os.path.join(images_path, target['filename']): {
                'ori': torch.tensor(target[ori_name]),
                'pos': torch.tensor(target[pos_name])
            } for target in target_list
        }

        # Sort images by filename for consistency, especially in video sequences
        self.data = dict(sorted(self.data.items(), key=get_image_number))

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image and its corresponding pose data
        image_path, target = copy.deepcopy(list(self.data.items())[idx])
        image = Image.open(image_path).convert("RGB")

        # Apply custom rotation transform if defined
        if self.rot_transform:
            image, target = self.rot_transform(image, target)

        # Store original image after custom rotation to ensure a matching between the true pose and the image
        original_image = torch.tensor(np.array(image))

        # Apply standard image transformation if defined
        if self.transform:
            image = self.transform(image)

        img = {
            # Store original image here for faster GUI (load the image twice would be too long)
            'original': original_image,
            'torch': image,
            'path': image_path
        }

        if self.spe_utils.keypoints is not None:
            target['keypoints'] = torch.tensor(self.spe_utils.keypoints.create_keypoints2d(
                target['ori'], target['pos'])
            )
            target['bbox'] = torch.tensor(self.spe_utils.keypoints.create_bbox_from_keypoints(
                target['keypoints'].numpy())
            )

        # Encode orientation for classification tasks
        if self.spe_utils.ori_mode == 'classification':
            target['ori_soft'] = torch.tensor(self.spe_utils.orientation.encode(target['ori'].numpy()))
        if self.spe_utils.pos_mode == 'classification':
            target['pos_soft'] = torch.tensor(self.spe_utils.position.encode(target['pos'].numpy()))

        return img, target


def seed_worker(worker_id):
    """This function is used as the `worker_init_fn` parameter in `torch.utils.data.DataLoader` to set the random seed
    for each worker process in a way that ensures reproducibility"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
