import os

from src.config.train.config import load_config
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.modeling.model import import_model
from src.tools.utils import select_device, set_seed
from src.tools.evaluation import evaluation
from src.tools.utils import save_score_error
from src.spe.spe_torch import SPETorch


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
    batch_size = 32
    rot_augment = False
    other_augment = False
    shuffle = False
    data, split = load_dataset(spe_utils, cfg.DATA.PATH, batch_size, cfg.DATA.IMG_SIZE,
                               rot_augment, other_augment, shuffle, seed)

    manual_copy = False
    model, model_bit_width = import_model(
        data, cfg.MODEL.BACKBONE.NAME, cfg.MODEL.HEAD.NAME, params_path, bit_width_path,
        manual_copy, residual=cfg.MODEL.BACKBONE.RESIDUAL, quantization=cfg.MODEL.QUANTIZATION,
        ori_mode=cfg.MODEL.HEAD.ORI, n_ori_bins=spe_utils.orientation.n_bins,
        pos_mode=cfg.MODEL.HEAD.POS, n_pos_bins=spe_utils.position.n_bins,
    )

    print(f"Number of trainable parameters in the model:"
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Evaluation
    spe_model = SPETorch(model, device, spe_utils)
    score, error = evaluation(spe_model, data, spe_utils, split['eval'])
    save_score_error(score, error, path=os.path.join('experiments', 'eval', exp_dir), name='eval.xlsx')


if __name__ == '__main__':
    main()
