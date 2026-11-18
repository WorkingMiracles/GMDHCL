"""Train GMDHCL with explicit seeds and validation-only model selection."""
import argparse
import csv
import gc
import io
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from run_dropout_stability import ROOT, train_one
from utils.atomic_io import atomic_bytes, read_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model', '-m', choices=['GMDHCL'], default='GMDHCL')
    parser.add_argument('--dataset', '-d', choices=['baby', 'sports', 'clothing', 'tiktok'], required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data',
                        help='Directory containing the dataset subdirectories')
    parser.add_argument('--config', type=Path, help='Optional private YAML overrides; no configuration file is required')
    parser.add_argument('--seeds', '--seed', type=int, nargs='+', default=[999])
    parser.add_argument('--gpu-id', '--gpu_id', '-g', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, help='Override maximum epochs; short runs are not formal paper results')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/training')
    parser.add_argument('--run-id', default='gmdhcl_v1')
    return parser


def write_summary(run_root, dataset, seeds):
    rows = []
    for seed in seeds:
        manifest = read_json(run_root / dataset / ('seed_' + str(seed)) / 'training_manifest.json')
        if manifest['status'] != 'complete':
            raise ValueError('Only completed training runs may enter a summary')
        rows.append({'seed': seed, **manifest['test_at_best_validation']})
    metric_names = sorted(key for key in rows[0] if key != 'seed')
    for name, operation in [('mean', np.mean), ('std', lambda values: np.std(values, ddof=1))]:
        rows.append({'seed': name, **{key: float(operation([row[key] for row in rows[:len(seeds)]]))
                     if name == 'mean' or len(seeds) > 1 else '' for key in metric_names}})
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=['seed'] + metric_names)
    writer.writeheader()
    writer.writerows(rows)
    target = run_root / dataset / 'summary.csv'
    atomic_bytes(target, buffer.getvalue().encode('utf-8'))
    print('Saved test-at-best-validation summary:', target, flush=True)
    return target


def main(argv=None):
    args = build_parser().parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        raise ValueError('Seeds must be unique integers in [0, 2**32)')
    if Path(args.run_id).name != args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id) or not args.run_id:
        raise ValueError('run-id must contain letters, numbers, underscores or hyphens')
    overrides = {}
    if args.config:
        with args.config.open(encoding='utf-8') as stream:
            overrides = yaml.safe_load(stream) or {}
        if not isinstance(overrides, dict):
            raise ValueError('Configuration overrides must be a YAML mapping')
    overrides['use_gpu'] = not args.cpu
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError('epochs must be positive')
        overrides['epochs'] = args.epochs
    args.data_root = str(args.data_root.resolve())
    run_root = args.output_root.resolve() / args.run_id
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    spec = {'model_overrides': overrides, 'diagnostic': False}
    for seed in args.seeds:
        train_one(spec, args, run_root, seed)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_summary(run_root, args.dataset, args.seeds)


if __name__ == '__main__':
    main()
