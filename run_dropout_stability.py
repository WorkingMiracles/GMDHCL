"""R2.2 runner: explicit smoke/train/collect/summarize, never an implicit grid search."""
import argparse
import copy
import gc
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
import yaml

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.utils import init_seed
from utils.stability_checkpoint import (plain, stable_digest, file_hash, resolve_config,
    runtime_info, atomic_json, read_checkpoint, load_checkpoint, validate_training_manifest)
from utils.stability_sampling import collect_checkpoint, write_collection
from utils.stability_statistics import summarize_collections

ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = {
    'experiment_id': 'gmdhcl_stability_v1',
    'datasets': ['baby', 'sports', 'clothing'],
    'train_seeds': [999, 42, 123, 2024, 2025, 2026],
    'master_seed': 20260911,
    'num_masks': 20,
    'output_root': 'experiments/dropout_stability',
    'model_overrides': {},
    'smoke': {'train_batches': 2, 'train_batch_size': 32, 'eval_users': 64, 'num_masks': 4},
}
MODEL_SOURCES = ('models/gmdhcl.py', 'models/graph_transformer.py', 'common/abstract_recommender.py',
                 'utils/dataset.py', 'utils/dataloader.py', 'common/trainer.py', 'utils/utils.py',
                 'utils/configurator.py')
SAMPLING_SOURCES = ('run_dropout_stability.py', 'utils/stability_checkpoint.py',
                    'utils/stability_sampling.py', 'utils/stability_statistics.py')


class LimitedBatches:
    """Smoke-only training limiter. Does not alter the full training graph."""
    def __init__(self, loader, limit):
        if limit < 1:
            raise ValueError('Batch limit must be positive')
        self.loader, self.limit = loader, int(limit)

    def __iter__(self):
        if hasattr(self.loader, 'pr'):
            self.loader.pr = self.loader.inter_pr = 0
        return itertools.islice(iter(self.loader), self.limit)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subs = parser.add_subparsers(dest='command', required=True)
    for command in ('smoke', 'train', 'collect', 'summarize'):
        p = subs.add_parser(command, allow_abbrev=False)
        p.add_argument('--config', help='Optional private YAML overrides; built-in defaults are used otherwise')
        p.add_argument('--run-id')
        if command != 'summarize':
            p.add_argument('--data-root', help='Directory containing baby/sports/clothing subdirectories')
            p.add_argument('--gpu-id', type=int, default=0)
            p.add_argument('--cpu', action='store_true', help='Run the diagnostic on CPU')
        p.add_argument('--dataset', choices=['baby', 'sports', 'clothing'],
                       default='baby' if command == 'smoke' else None,
                       required=command in ('train', 'collect'))
        if command in ('train', 'collect'):
            selection = p.add_mutually_exclusive_group(required=True)
            selection.add_argument('--train-seed', type=int)
            selection.add_argument('--all-seeds', action='store_true')
        elif command == 'smoke':
            p.add_argument('--train-seed', type=int, default=999)
        else:
            p.add_argument('--allow-partial', action='store_true')
            p.add_argument('--allow-smoke', action='store_true')
    return parser


def load_spec(args):
    spec = copy.deepcopy(DEFAULT_SPEC)
    if args.config:
        with Path(args.config).resolve().open(encoding='utf-8') as stream:
            overrides = yaml.safe_load(stream) or {}
        if not isinstance(overrides, dict):
            raise ValueError('Experiment overrides must be a YAML mapping')
        spec.update(overrides)
    seeds = spec['train_seeds']
    if len(set(seeds)) != len(seeds) or not all(isinstance(s, int) and 0 <= s < 2**32 for s in seeds):
        raise ValueError('Training seeds must be unique integers in [0, 2**32)')
    if not isinstance(spec['num_masks'], int) or spec['num_masks'] < 2:
        raise ValueError('num_masks must be at least two')
    if not set(spec['datasets']).issubset({'baby', 'sports', 'clothing'}) or not spec['datasets']:
        raise ValueError('Unsupported datasets')
    run_id = args.run_id or ('r2_2_smoke' if args.command == 'smoke' else spec['experiment_id'])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('run-id must contain only letters, numbers, underscores or hyphens')
    base = Path(spec['output_root'])
    if not base.is_absolute():
        base = ROOT / base
    return spec, base.resolve() / run_id


def build_config(spec, dataset, seed, gpu_id, data_root, smoke=False, saved=None):
    # Config uses built-in parameters and optional private overrides.
    overrides = copy.deepcopy(saved if saved is not None else spec.get('model_overrides', {}))
    data_base = Path(data_root).resolve() if data_root else ROOT / 'data'
    if saved is None or data_root:
        overrides['data_path'] = str(data_base) + os.sep
    overrides['gpu_id'] = gpu_id
    overrides.setdefault('use_gpu', True)
    config = Config('GMDHCL', dataset, overrides)
    config.final_config_dict = resolve_config(config.final_config_dict, seed)
    config['stability_run_mode'] = 'smoke_only' if smoke else 'formal'
    if smoke:
        config['epochs'] = 1
        config['train_batch_size'] = int(spec['smoke']['train_batch_size'])
        config['eval_batch_size'] = int(spec['smoke']['eval_users'])
        config['eval_step'] = 1
        config['stability_smoke_train_batches'] = int(spec['smoke']['train_batches'])
        config['stability_smoke_eval_users'] = int(spec['smoke']['eval_users'])
    if spec.get('diagnostic', True) and (not config['use_gmd'] or config['use_spectral_perturb']):
        raise ValueError('Requires use_gmd=True and use_spectral_perturb=False')
    if config['use_gpu']:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select --cpu to run on CPU')
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError('Invalid GPU index')
        torch.cuda.set_device(gpu_id)
    return config


def datasets_and_fingerprints(config):
    full = RecDataset(config)
    splits = full.split()
    for part in splits:
        part.inter_num = len(part)  # Do not rely on __str__ side effects.
    train = splits[0]
    base = Path(config['data_path']) / config['dataset']
    for key in ('vision_feature_file', 'text_feature_file'):
        if not (base / config[key]).is_file():
            raise FileNotFoundError('GMDHCL requires both visual and textual item feature files; missing: ' + str(base / config[key]))
    fingerprints = {name: file_hash(base / config[key]) for name, key in
                    (('interactions', 'inter_file_name'), ('vision', 'vision_feature_file'), ('text', 'text_feature_file'))}
    pairs = train.df[[train.uid_field, train.iid_field]].to_numpy(dtype=np.int64)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    fingerprints['training_pairs'] = hashlib.sha256(pairs.tobytes()).hexdigest()
    fingerprints['n_users'], fingerprints['n_items'] = int(full.user_num), int(full.item_num)
    fingerprints['model_code'] = {name: file_hash(ROOT / name) for name in MODEL_SOURCES}
    return splits, fingerprints


def loaders(config, splits, smoke=False):
    train, valid, test = splits
    train_loader = TrainDataLoader(config, train, batch_size=config['train_batch_size'], shuffle=True)
    train_loader.pretrain_setup()
    eval_loaders = []
    for part in (valid, test):
        if smoke:
            user_ids = np.sort(part.df[part.uid_field].unique())[:config['stability_smoke_eval_users']]
            part = part.copy(part.df[part.df[part.uid_field].isin(user_ids)].copy())
            part.inter_num = len(part)
        eval_loaders.append(EvalDataLoader(config, part, additional_dataset=train,
                           batch_size=config['eval_batch_size']))
    return train_loader, eval_loaders[0], eval_loaders[1]


def run_seed_dir(run_root, dataset, seed):
    return run_root / dataset / ('seed_' + str(seed))


def train_one(spec, args, run_root, seed, smoke=False):
    from models.gmdhcl import GMDHCL
    from common.trainer import Trainer
    config = build_config(spec, args.dataset, seed, args.gpu_id, args.data_root, smoke=smoke)
    init_seed(seed)
    splits, fingerprints = datasets_and_fingerprints(config)
    output = run_seed_dir(run_root, args.dataset, seed)
    config['stability_checkpoint_path'] = str(output / 'best.pt')
    config['stability_fingerprints'] = fingerprints
    config_digest = stable_digest(config.final_config_dict)
    if output.exists():
        manifest_path = output / 'training_manifest.json'
        if manifest_path.exists():
            with manifest_path.open(encoding='utf-8') as stream:
                previous = json.load(stream)
            if (previous['status'] == 'complete' and previous['config_digest'] == config_digest and
                    file_hash(output / 'best.pt') == previous['checkpoint_sha256']):
                print('Verified completed training; skipping:', output, flush=True)
                validate_training_manifest(output)
                return output / 'best.pt'
        raise FileExistsError('Existing incomplete/incompatible run; choose a new --run-id: ' + str(output))
    output.mkdir(parents=True)
    manifest = {'status': 'running', 'dataset': args.dataset, 'train_seed': seed,
                'run_mode': config['stability_run_mode'], 'config_digest': config_digest,
                'resolved_config': plain(config.final_config_dict), 'fingerprints': fingerprints,
                'runtime': runtime_info()}
    atomic_json(output / 'training_manifest.json', manifest)
    handler = logging.FileHandler(output / 'training.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger().addHandler(handler)
    started = time.perf_counter()
    if config['device'].type == 'cuda':
        torch.cuda.reset_peak_memory_stats(config['device'])
    try:
        train, valid, test = loaders(config, splits, smoke)
        model = GMDHCL(config, train).to(config['device'])
        trainer = Trainer(config, model)
        training_data = LimitedBatches(train, config['stability_smoke_train_batches']) if smoke else train
        print('Training', args.dataset, 'seed', seed, 'mode', config['stability_run_mode'], flush=True)
        score, valid_result, test_result = trainer.fit(training_data, valid_data=valid, test_data=test, saved=True)
        checkpoint = output / 'best.pt'
        if not checkpoint.exists():
            raise RuntimeError('Training did not produce a validation-best checkpoint')
        manifest.update(status='complete', best_valid_score=float(score), best_valid_result=valid_result,
                        test_at_best_validation=test_result, elapsed_seconds=time.perf_counter() - started,
                        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(config['device'])) if config['device'].type == 'cuda' else 0,
                        checkpoint_sha256=file_hash(checkpoint))
        atomic_json(output / 'training_manifest.json', manifest)
        print('Saved validation-best checkpoint:', checkpoint, flush=True)
        return checkpoint
    except Exception as error:
        manifest.update(status='failed', error=repr(error), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / 'training_manifest.json', manifest)
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def collect_one(spec, args, run_root, seed, allow_smoke=False):
    from models.gmdhcl import GMDHCL
    output = run_seed_dir(run_root, args.dataset, seed)
    checkpoint = output / 'best.pt'
    training_manifest, payload = validate_training_manifest(output)
    smoke = payload['run_mode'] == 'smoke_only'
    if smoke and not allow_smoke:
        raise ValueError('Smoke checkpoint cannot enter formal collection')
    if payload['train_seed'] != seed or payload['resolved_config']['dataset'] != args.dataset:
        raise ValueError('Checkpoint seed/dataset mismatch')
    config = build_config(spec, args.dataset, seed, args.gpu_id, args.data_root,
                          smoke=smoke, saved=payload['resolved_config'])
    splits, fingerprints = datasets_and_fingerprints(config)
    train = TrainDataLoader(config, splits[0], batch_size=config['train_batch_size'], shuffle=False)
    model = GMDHCL(config, train).to(config['device'])
    load_checkpoint(checkpoint, model, expected_fingerprints=fingerprints)
    num_masks = int(spec['smoke']['num_masks'] if smoke else spec['num_masks'])
    metadata = {'dataset': args.dataset, 'train_seed': seed, 'num_masks': num_masks,
                'master_seed': int(spec['master_seed']), 'run_mode': payload['run_mode'],
                'checkpoint_sha256': file_hash(checkpoint), 'best_epoch': payload['best_epoch'],
                'resolved_config': payload['resolved_config'], 'fingerprints': fingerprints,
                'sampling_code': {name: file_hash(ROOT / name) for name in SAMPLING_SOURCES},
                'runtime': runtime_info(), 'measurement': 'frozen checkpoint, full graph, mge + geometric_fusion only'}
    collection = output / 'collection'
    if (collection / 'manifest.json').exists():
        with (collection / 'manifest.json').open(encoding='utf-8') as stream:
            old = json.load(stream)
        if all(old.get(k) == v for k, v in metadata.items()) and old.get('status') == 'complete' and \
                file_hash(collection / 'measurements.npz') == old['measurements_sha256']:
            print('Verified completed collection; skipping:', collection, flush=True)
            return collection
        raise FileExistsError('Existing incompatible collection: ' + str(collection))
    started = time.perf_counter()
    result = collect_checkpoint(model, dataset_name=args.dataset, train_seed=seed,
        master_seed=spec['master_seed'], num_masks=num_masks,
        progress=lambda k, total: print('Mask {}/{}: {} seed {}'.format(k, total, args.dataset, seed), flush=True))
    metadata['elapsed_seconds'] = time.perf_counter() - started
    write_collection(collection, result, metadata)
    print('Saved collection:', collection, flush=True)
    return collection


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Resolve user paths before changing cwd for the existing configurator.
    if args.config:
        args.config = str(Path(args.config).resolve())
    if getattr(args, 'data_root', None):
        args.data_root = str(Path(args.data_root).resolve())
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    spec, run_root = load_spec(args)
    if getattr(args, 'cpu', False):
        spec.setdefault('model_overrides', {})['use_gpu'] = False
    if args.dataset and args.dataset not in spec['datasets']:
        raise ValueError('Dataset is not in the experiment specification')
    if args.command == 'smoke':
        train_one(spec, args, run_root, args.train_seed, smoke=True)
        gc.collect()
        torch.cuda.empty_cache()
        collection = collect_one(spec, args, run_root, args.train_seed, allow_smoke=True)
        report = summarize_collections([collection], run_seed_dir(run_root, args.dataset, args.train_seed) / 'summary',
                                       expected_seeds=[args.train_seed])
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    elif args.command in ('train', 'collect'):
        seeds = spec['train_seeds'] if args.all_seeds else [args.train_seed]
        if not set(seeds).issubset(spec['train_seeds']):
            raise ValueError('Training seed is not in the predeclared experiment specification')
        for seed in seeds:
            if args.command == 'train':
                train_one(spec, args, run_root, seed)
            else:
                collect_one(spec, args, run_root, seed)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        datasets = [args.dataset] if args.dataset else spec['datasets']
        directories = []
        for dataset in datasets:
            found = sorted((run_root / dataset).glob('seed_*/collection/manifest.json'))
            if not found and not args.allow_partial:
                raise ValueError('No collections for required dataset: ' + dataset)
            for path in found:
                with path.open(encoding='utf-8') as stream:
                    meta = json.load(stream)
                if meta['run_mode'] != 'formal' and not args.allow_smoke:
                    raise ValueError('Use --allow-smoke only for exploratory smoke summaries')
                directories.append(path.parent)
        report = summarize_collections(directories, run_root / 'summary', expected_seeds=spec['train_seeds'],
                                       allow_partial=args.allow_partial)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
