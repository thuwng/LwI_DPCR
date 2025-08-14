import json
import argparse
import torch
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    # Thêm các tham số mặc định cho GroundMetric nếu không có trong JSON
    default_params = {
        'ground_metric': args.get('ground_metric', 'cosine'),  # Mặc định là cosine
        'ground_metric_normalize': args.get('ground_metric_normalize', 'none'),  # Mặc định không normalize
        'reg': args.get('reg', 1.0),  # Hệ số regularization mặc định
        'ground_metric_eff': args.get('ground_metric_eff', False),  # Mặc định không dùng memory efficiency
        'not_squared': args.get('not_squared', False),  # Mặc định squared = True
        'clip_max': args.get('clip_max', 100.0),  # Giá trị tối đa mặc định
        'clip_min': args.get('clip_min', 0.0),  # Giá trị tối thiểu mặc định
        'dist_normalize': args.get('dist_normalize', False),  # Mặc định không normalize distance
        'geom_ensemble_type': args.get('geom_ensemble_type', 'none'),  # Mặc định không ensemble
        'normalize_wts': args.get('normalize_wts', False),  # Mặc định không normalize weights
        'act_num_samples': args.get('act_num_samples', 1.0),  # Mặc định 1.0
        'debug': args.get('debug', False),  # Mặc định không debug
        'clip_gm': args.get('clip_gm', True),  # Mặc định áp dụng clipping
        'device': args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')  # Thiết bị mặc định
    }
    args.update(default_params)  # Cập nhật args với các giá trị mặc định

    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')

    return parser

if __name__ == '__main__':
    main()