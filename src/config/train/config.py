import os.path
from yacs.config import CfgNode as ConfigurationNode

_C = ConfigurationNode()

_C.MODEL = ConfigurationNode()
_C.MODEL.PRETRAINED_PATH = 'models/fp32_12bins_model.pt'
_C.MODEL.MANUAL_COPY = True
_C.MODEL.QUANTIZATION = False

_C.MODEL.BACKBONE = ConfigurationNode()
_C.MODEL.BACKBONE.NAME = 'mobilenet_v2_pytorch'
_C.MODEL.BACKBONE.RESIDUAL = True

_C.MODEL.HEAD = ConfigurationNode()
_C.MODEL.HEAD.NAME = 'ursonet_pytorch'
_C.MODEL.HEAD.ORI = 'classification'
_C.MODEL.HEAD.POS = 'regression'
_C.MODEL.HEAD.N_ORI_BINS_PER_DIM = 12
_C.MODEL.HEAD.N_POS_BINS_PER_DIM = 10
_C.MODEL.HEAD.ORI_DELETE_UNUSED_BINS = False
_C.MODEL.HEAD.KEYPOINTS_PATH = 'models/3d_models/tangoPoints.mat'

_C.DATA = ConfigurationNode()
_C.DATA.BATCH_SIZE = 8
_C.DATA.PATH = '../datasets/speed'
_C.DATA.IMG_SIZE = (240, 384)
_C.DATA.ORI_SMOOTH_FACTOR = 3
_C.DATA.POS_SMOOTH_FACTOR = 100
_C.DATA.ROT_AUGMENT = True
_C.DATA.OTHER_AUGMENT = True
_C.DATA.SHUFFLE = True

_C.TRAIN = ConfigurationNode()
_C.TRAIN.N_EPOCH = 2
_C.TRAIN.LR = 0.01
_C.TRAIN.OPTIM = 'SGD'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.DECAY = 0.0
_C.TRAIN.SCHEDULER = 'MultiStepLR'
_C.TRAIN.MILESTONES = (7, 20)
_C.TRAIN.GAMMA = 0.1
_C.TRAIN.CLIP_BATCHNORM = False


def load_config(path=None):
    """Load configuration for FINN build. Optionally load parameters from a YAML file"""
    if path is not None:
        assert os.path.isfile(path), f'File {path} does not exist'
        _C.merge_from_file(path)
    # assert _C.MODEL.BACKBONE.NAME in ('mobilenet_v2_brevitas', 'mobilenet_v2_pytorch', 'small_brevitas')
    # assert _C.MODEL.HEAD.NAME in ('ursonet_brevitas', 'ursonet_pytorch')
    assert _C.MODEL.HEAD.ORI in ('classification', 'regression', 'keypoints')
    assert _C.MODEL.HEAD.POS in ('classification', 'regression', 'keypoints')
    # Check if one is 'keypoints', then the other must be 'keypoints' too
    if _C.MODEL.HEAD.ORI == 'keypoints' or _C.MODEL.HEAD.POS == 'keypoints':
        assert _C.MODEL.HEAD.ORI == 'keypoints' and _C.MODEL.HEAD.POS == 'keypoints', \
            "Both ORI and POS must be 'keypoints' if one is 'keypoints'"

    return _C.clone()


def save_config(config, path=None):
    """Save the configuration to a YAML file"""
    assert os.path.exists(os.path.dirname(path)), f'Path {path} does not exist'
    config.dump(stream=open(path, 'w'))
