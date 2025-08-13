import os.path
from yacs.config import CfgNode as CN

# Create the root configuration node
_C = CN()

# DATA configuration
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 32
_C.DATA.OTHER_AUGMENT = True
_C.DATA.ROT_AUGMENT = True
_C.DATA.SHUFFLE = True

# MODEL configuration
_C.MODEL = CN()
_C.MODEL.PRETRAINED_PATH = "models/murso_fp32_speed"
_C.MODEL.QUANTIZATION = True

# TRAIN configuration
_C.TRAIN = CN()
_C.TRAIN.QAT = True
_C.TRAIN.DECAY = 0.0
_C.TRAIN.GAMMA = 0.1
_C.TRAIN.LR = 0.01
_C.TRAIN.MILESTONES = [7, 15]
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.N_EPOCH = 1
_C.TRAIN.OPTIM = "SGD"
_C.TRAIN.SCHEDULER = "MultiStepLR"
_C.TRAIN.CLIP_BATCHNORM = False

# COMPILE configuration
_C.COMPILE = CN()
_C.COMPILE.RUN_AUTOSCHEDULING = True
_C.COMPILE.N_TRIALS = 1000
_C.COMPILE.SCHEDULING_PATH = "experiments/build/tvm/scheduling/scheduling.json"
_C.COMPILE.BOARD = "Ultra96"



def load_config(path=None):
    """Load configuration for FINN build. Optionally load parameters from a YAML file"""
    if path is not None:
        assert os.path.isfile(path), f'File {path} does not exist'
        _C.merge_from_file(path)
    return _C.clone()


def save_config(config, path=None):
    """Save the configuration to a YAML file"""
    assert os.path.exists(os.path.dirname(path)), f'Path {path} does not exists'
    config.dump(stream=open(path, 'w'))
