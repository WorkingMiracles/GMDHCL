"""Offline repeated-mask diagnostics using the original mge dropout call."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from utils.stability_checkpoint import model_state_hash, atomic_json
from utils.stability_statistics import degree_groups, FusionMoments, verify_weight_transition

PAIR_FIELDS = ('u_single', 'u_used', 'reliability_raw', 'weight_before',
               'weight_after', 'unique_norm', 'clean_norm')
NODE_FIELDS = PAIR_FIELDS + ('lambda', 'cosine_vt')


def make_mask_seed(master_seed, dataset, train_seed, mask_index, modality):
    if modality not in ('v', 't'):
        raise ValueError('Expected v or t modality')
    key = json.dumps([int(master_seed), str(dataset), int(train_seed), int(mask_index), modality])
    digest = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'little') & ((1 << 63) - 1)


@torch.no_grad()
def sample_mge(model, modality, mask_seed):
    if modality not in ('v', 't') or not model.use_gmd:
        raise ValueError('Expected v/t modality and use_gmd=True')
    if model.use_spectral_perturb:
        raise ValueError('R2.2 version 1 supports ordinary feature dropout, not spectral dropout')
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(int(mask_seed))
        result = model.mge(True, modality)
    if len(result) == 2:
        return result[0], result[1], None
    if len(result) == 3:
        return result
    raise RuntimeError('Unexpected mge return structure')


@torch.no_grad()
def collect_checkpoint(model, *, dataset_name, train_seed, master_seed, num_masks, progress=None):
    if num_masks < 2:
        raise ValueError('At least two masks required')
    if not model.use_gmd or model.use_spectral_perturb or model.cl_layer < 1:
        raise ValueError('Requires GMD, ordinary dropout, and positive propagation depth')
    if not (0 <= model.gmd_drop_rate < 1):
        raise ValueError('Dropout rate must be in [0,1)')
    modes = [(module, module.training) for module in model.modules()]
    before = model_state_hash(model)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    graph = model.adj.coalesce()
    degree = np.bincount(graph.indices()[1].cpu().numpy(),
                         weights=graph.values().cpu().numpy(), minlength=model.n_items)
    groups = degree_groups(degree)
    result = {key: np.empty((num_masks, model.n_items, 2) if key in PAIR_FIELDS else
                           (num_masks, model.n_items), dtype=np.float32) for key in NODE_FIELDS}
    result['tau'] = np.empty(num_masks, dtype=np.float32)
    for modality in ('v', 't'):
        result['mask_seed_' + modality] = np.array([
            make_mask_seed(master_seed, dataset_name, train_seed, k, modality)
            for k in range(num_masks)], dtype=np.int64)
    all_seeds = np.concatenate([result['mask_seed_v'], result['mask_seed_t']])
    if len(np.unique(all_seeds)) != 2 * num_masks:
        raise RuntimeError('Mask seed collision')
    moments = FusionMoments()
    item_slice = slice(model.n_users, model.n_users + model.n_items)
    model.eval()
    try:
        for k in range(num_masks):
            v, vc, vm = sample_mge(model, 'v', result['mask_seed_v'][k])
            t, tc, tm = sample_mge(model, 't', result['mask_seed_t'][k])
            fused, diagnostics = model.geometric_fusion(v, vc, t, tc, vm, tm, return_diagnostics=True)
            for key in NODE_FIELDS:
                value = diagnostics[key][item_slice].detach().cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite diagnostic field: ' + key)
                result[key][k] = value
            tau = float(diagnostics['tau'].item())
            if not np.isfinite(tau) or tau <= 0:
                raise ValueError('Invalid uncertainty temperature')
            result['tau'][k] = tau
            moments.update(fused[item_slice].detach().cpu().numpy())
            if progress:
                progress(k + 1, num_masks)
    finally:
        for module, training in modes:
            module.training = training
    state_unchanged = before == model_state_hash(model)
    rng_unchanged = torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng:
        rng_unchanged = rng_unchanged and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    if not state_unchanged or not rng_unchanged:
        raise RuntimeError('Collection changed model/graph state or global RNG')
    if not np.allclose(result['lambda'], result['lambda'][0], rtol=0, atol=0):
        raise RuntimeError('Item degree weights changed between masks')
    transition_checks = verify_weight_transition(result['weight_before'], result['weight_after'], result['lambda'])
    if not np.allclose(result['clean_norm'], result['clean_norm'][0], rtol=1e-6, atol=1e-7):
        raise RuntimeError('Clean representation changed across masks')
    result.update(moments.finalize())
    result.update(item_id=np.arange(model.n_items, dtype=np.int64), train_degree=degree.astype(np.int64), group_code=groups)
    result['quality_checks'] = {'state_unchanged': True, 'rng_unchanged': True,
                                'variance_identity': True, 'clean_norm_invariant': True,
                                'all_finite': True, 'completed_masks': num_masks}
    result['quality_checks'].update(transition_checks)
    return result


def write_collection(output_dir, result, metadata):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists() or (output / 'measurements.npz').exists():
        raise FileExistsError('Collection exists; refusing to overwrite ' + str(output))
    temporary = output / 'measurements.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    os.replace(str(temporary), str(output / 'measurements.npz'))
    from utils.stability_checkpoint import file_hash
    manifest = dict(metadata, format_version=1, status='complete',
                    modality_order=['visual', 'text'], pair_axes=['mask', 'item', 'modality'],
                    quality_checks=result['quality_checks'], measurements_sha256=file_hash(output / 'measurements.npz'))
    atomic_json(output / 'manifest.json', manifest)
