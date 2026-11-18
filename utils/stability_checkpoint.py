"""Versioned, fingerprinted checkpoints for offline R2.2 diagnostics."""
import copy
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Cannot serialize configuration value: ' + str(type(value)))


def stable_digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor):
    digest = hashlib.sha256()
    tensor = tensor.detach()
    digest.update(str((tuple(tensor.shape), tensor.dtype, tensor.layout)).encode())
    if tensor.layout == torch.sparse_coo:
        tensor = tensor.coalesce()
        digest.update(tensor_hash(tensor.indices()).encode())
        digest.update(tensor_hash(tensor.values()).encode())
    else:
        digest.update(tensor.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_state_hash(model):
    values = dict(model.state_dict())
    for name in ('adj', 'norm_adj', 'num_inters', 'norm_adj_iu'):
        if hasattr(model, name):
            values['unregistered_graph:' + name] = getattr(model, name)
    return stable_digest({name: tensor_hash(value) for name, value in sorted(values.items())})


def resolve_config(config, train_seed):
    resolved = copy.deepcopy(config)
    for name in resolved.get('hyper_parameters', []):
        if name == 'seed':
            continue
        value = resolved.get(name)
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError('Choose exactly one value for search parameter: ' + name)
            resolved[name] = value[0]
    resolved['seed'] = int(train_seed)
    resolved['hyper_parameters'] = []
    return resolved


def runtime_info():
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': str(np.__version__), 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(str(temporary), str(path))


def save_checkpoint(path, model, config, fingerprints, *, epoch, valid_result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': FORMAT_VERSION,
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'resolved_config': plain(config), 'fingerprints': plain(fingerprints),
        'train_seed': int(config['seed']), 'best_epoch': int(epoch),
        'selection_metric': config.get('valid_metric', 'recall@20'),
        'best_valid_result': plain(valid_result), 'runtime': runtime_info(),
        'run_mode': config.get('stability_run_mode', 'formal'),
        'evaluation_protocol': 'original validation and test calls each evaluation epoch; selection uses validation only',
    }
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def read_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format')
    return payload


def load_checkpoint(path, model, *, expected_fingerprints):
    payload = read_checkpoint(path)
    if payload['fingerprints'] != plain(expected_fingerprints):
        raise ValueError('Checkpoint data/features/model-code fingerprints do not match')
    state = payload['model_state_dict']
    has_degree = hasattr(model, 'item_degree_norm')
    if has_degree != ('item_degree_norm' in state):
        raise ValueError('Checkpoint geometric-fusion branch does not match model configuration')
    if has_degree and not torch.equal(model.item_degree_norm.detach().cpu(), state['item_degree_norm'].cpu()):
        raise ValueError('Checkpoint item degrees do not match reconstructed training graph')
    model.load_state_dict(state, strict=True)
    return payload


def validate_training_manifest(run_dir):
    """A surviving best.pt is not proof that the training run completed."""
    root = Path(run_dir)
    manifest_path = root / 'training_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Training completion manifest is missing: ' + str(root))
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    if manifest.get('status') != 'complete':
        raise ValueError('Training is not complete: ' + str(root))
    if file_hash(root / 'best.pt') != manifest.get('checkpoint_sha256'):
        raise ValueError('Training checkpoint checksum mismatch')
    payload = read_checkpoint(root / 'best.pt')
    cfg = payload['resolved_config']
    if (stable_digest(cfg) != manifest.get('config_digest') or
            cfg != manifest.get('resolved_config') or
            payload['fingerprints'] != manifest.get('fingerprints') or
            payload['train_seed'] != manifest.get('train_seed') or
            cfg.get('dataset') != manifest.get('dataset') or
            payload['run_mode'] != manifest.get('run_mode')):
        raise ValueError('Training manifest does not describe this checkpoint')
    return manifest, payload
