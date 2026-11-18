"""Pure CPU statistics. Axes are mask/seed, item, modality (visual,text)."""
import numpy as np

GROUP_LABELS = ('zero', 'cold_1_5', 'medium_6_20', 'warm_gt20')
MEASURES = ('u_single', 'u_used', 'reliability_raw', 'weight_before', 'weight_after', 'unique_norm', 'clean_norm')


def _repetitions(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 2 or array.shape[0] < 2:
        raise ValueError('Expected [at least 2 repetitions, items, 2 modalities]')
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite raw measurements')
    return array


def mask_statistics(values):
    array = _repetitions(values)
    return array.mean(axis=0), array.std(axis=0, ddof=1)


def seed_statistics(mask_means):
    return _repetitions(mask_means).std(axis=0, ddof=1)


def verify_weight_transition(before, after, lam):
    """Check the affine weight rule with a dtype-derived rounding bound.

    If each rounded output has error <= delta, its sample SD error is bounded
    by sqrt(K/(K-1))*delta. Thus |Var(y)-Var(x)| <= 2*SD(x)*b+b**2.
    This does not alter or round any measurements used for reporting.
    """
    dtype = np.result_type(np.asarray(before).dtype, np.asarray(after).dtype)
    before, after = _repetitions(before), _repetitions(after)
    lam = np.asarray(lam, dtype=np.float64)
    if after.shape != before.shape or lam.shape != before.shape[:2]:
        raise ValueError('Weight transition shape mismatch')
    if not np.isfinite(lam).all() or np.any(lam < 0) or np.any(lam > 1) or not np.array_equal(lam, np.broadcast_to(lam[0], lam.shape)):
        raise ValueError('Lambda must be fixed across masks and in [0,1]')
    expected = lam[:, :, None] * before + (1 - lam[:, :, None]) * .5
    delta = 4 * np.finfo(dtype).eps * max(1., float(np.abs(expected).max()))
    weight_error = np.abs(after - expected)
    if np.any(weight_error > delta):
        raise ValueError('Cold/warm affine weight identity failed beyond floating-point bound')
    predicted = lam[0, :, None] ** 2 * before.var(0, ddof=1)
    observed = after.var(0, ddof=1)
    b = np.sqrt(before.shape[0] / (before.shape[0] - 1)) * delta
    bound = 2 * np.sqrt(predicted) * b + b * b + 32 * np.finfo(np.float64).eps
    error = np.abs(observed - predicted)
    if np.any(error > bound):
        raise ValueError('Cold/warm variance identity failed beyond floating-point bound')
    return {'variance_identity': True, 'max_weight_roundoff': float(weight_error.max()),
            'weight_roundoff_bound': float(delta), 'max_variance_roundoff': float(error.max()),
            'variance_check': 'affine identity plus dtype-derived sample-variance error bound'}


def degree_groups(degrees):
    degree = np.asarray(degrees)
    if not np.isfinite(degree).all() or np.any(degree < 0) or np.any(degree != np.floor(degree)):
        raise ValueError('Training degrees must be finite nonnegative integers')
    return np.select([degree == 0, degree <= 5, degree <= 20], [0, 1, 2], default=3).astype(np.int8)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite statistics')
    if values.size == 0:
        return {'count': 0, 'median': None, 'q25': None, 'q75': None, 'p90': None}
    q25, median, q75, p90 = np.percentile(values, [25, 50, 75, 90])
    return dict(count=int(values.size), median=float(median), q25=float(q25), q75=float(q75), p90=float(p90))


class FusionMoments:
    """Welford accumulation: keep a mean vector and scalar M2 per item."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, vectors):
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite [items, features] fusion vectors')
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.m2 = np.zeros(x.shape[0], dtype=np.float64)
        if x.shape != self.mean.shape:
            raise ValueError('Fusion shape changed between masks')
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += np.sum(delta * (x - self.mean), axis=-1)

    def finalize(self):
        if self.count < 2:
            raise ValueError('At least two masks required')
        absolute = np.sqrt(np.maximum(self.m2 / (self.count - 1), 0))
        norm = np.linalg.norm(self.mean, axis=-1)
        return {'fusion_absolute_drift': absolute,
                'fusion_relative_drift': absolute / (norm + 1e-8),
                'fusion_mean_norm': norm, 'fusion_near_zero': norm <= 1e-8}


def validate_measurements(data, meta):
    """Verify the complete persisted schema and replayable mask protocol."""
    from utils.stability_sampling import make_mask_seed
    k, n = meta.get('num_masks'), meta.get('fingerprints', {}).get('n_items')
    if not isinstance(k, int) or k < 2 or not isinstance(n, int) or n < 1:
        raise ValueError('Missing/invalid mask count or fingerprint item count')
    shapes = {name: (k, n, 2) for name in MEASURES}
    shapes.update({'lambda': (k, n), 'cosine_vt': (k, n), 'tau': (k,),
        'mask_seed_v': (k,), 'mask_seed_t': (k,), 'item_id': (n,), 'train_degree': (n,),
        'group_code': (n,), 'fusion_absolute_drift': (n,), 'fusion_relative_drift': (n,),
        'fusion_mean_norm': (n,), 'fusion_near_zero': (n,)})
    for name, shape in shapes.items():
        if name not in data or data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError('Missing/invalid field: ' + name)
    for name in ('item_id', 'train_degree', 'group_code', 'mask_seed_v', 'mask_seed_t'):
        if data[name].dtype.kind not in ('i', 'u'):
            raise ValueError('Expected integer field: ' + name)
    if not np.array_equal(data['item_id'], np.arange(n)):
        raise ValueError('Item IDs must be canonical 0..n_items-1')
    if not np.array_equal(degree_groups(data['train_degree']), data['group_code']):
        raise ValueError('Incorrect degree groups')
    for name in MEASURES:
        if data[name].dtype.kind != 'f' or np.any(data[name] < 0):
            raise ValueError('Expected nonnegative floating-point field: ' + name)
        _repetitions(data[name])
    if np.any(data['tau'] <= 0) or np.any(np.abs(data['cosine_vt']) > 1 + 1e-6):
        raise ValueError('Invalid temperature or cosine similarity')
    for modality in ('v', 't'):
        expected = np.array([make_mask_seed(meta['master_seed'], meta['dataset'], meta['train_seed'], index, modality)
                             for index in range(k)], dtype=np.int64)
        if not np.array_equal(expected, data['mask_seed_' + modality]):
            raise ValueError('Saved mask seeds do not match the experiment protocol')
    if len(np.unique(np.concatenate([data['mask_seed_v'], data['mask_seed_t']]))) != 2 * k:
        raise ValueError('Duplicate mask seeds')
    verify_weight_transition(data['weight_before'], data['weight_after'], data['lambda'])
    if not np.allclose(data['clean_norm'], data['clean_norm'][0], rtol=1e-6, atol=1e-7):
        raise ValueError('Clean norms changed between masks')
    if data['fusion_near_zero'].dtype.kind != 'b' or not np.array_equal(data['fusion_near_zero'], data['fusion_mean_norm'] <= 1e-8):
        raise ValueError('Invalid fusion near-zero flags')
    for name in ('fusion_absolute_drift', 'fusion_relative_drift', 'fusion_mean_norm'):
        if np.any(data[name] < 0):
            raise ValueError('Negative fusion statistic')
    if not np.allclose(data['fusion_relative_drift'], data['fusion_absolute_drift'] / (data['fusion_mean_norm'] + 1e-8)):
        raise ValueError('Inconsistent relative fusion drift')


def summarize_collections(collection_dirs, output_dir, *, expected_seeds, allow_partial=False):
    """Validate all sources BEFORE emitting CSVs. No fabricated single-seed SD."""
    import csv
    import json
    from collections import defaultdict
    from pathlib import Path
    from utils.stability_checkpoint import file_hash, stable_digest, atomic_json

    loaded = defaultdict(list)
    seen = set()
    hashes = set()
    for directory in collection_dirs:
        directory = Path(directory)
        with (directory / 'manifest.json').open(encoding='utf-8') as stream:
            meta = json.load(stream)
        if meta.get('status') != 'complete' or meta.get('format_version') != 1:
            raise ValueError('Incomplete or unsupported collection: ' + str(directory))
        key = (meta['dataset'], int(meta['train_seed']))
        if key in seen or meta['checkpoint_sha256'] in hashes:
            raise ValueError('Duplicate training seed or reused checkpoint')
        seen.add(key)
        hashes.add(meta['checkpoint_sha256'])
        if file_hash(directory / 'measurements.npz') != meta['measurements_sha256']:
            raise ValueError('Raw measurement checksum mismatch')
        if not all(meta['quality_checks'].get(k) is True for k in
                   ('state_unchanged', 'rng_unchanged', 'variance_identity', 'clean_norm_invariant', 'all_finite')):
            raise ValueError('Collection quality checks did not pass')
        with np.load(directory / 'measurements.npz', allow_pickle=False) as raw:
            data = {name: raw[name] for name in raw.files}
        validate_measurements(data, meta)
        k, n = int(meta['num_masks']), len(data['item_id'])
        if meta['quality_checks']['completed_masks'] != k or len(np.unique(data['item_id'])) != n:
            raise ValueError('Incomplete masks or duplicate item IDs')
        for name in MEASURES:
            if data[name].shape != (k, n, 2):
                raise ValueError('Measurement shape mismatch: ' + name)
            _repetitions(data[name])
        for name, value in data.items():
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite field: ' + name)
        if not np.array_equal(degree_groups(data['train_degree']), data['group_code']):
            raise ValueError('Incorrect degree groups')
        loaded[key[0]].append((meta, data))
    if not loaded:
        raise ValueError('No completed collections found')

    report = {'status': 'complete_for_requested_inputs', 'datasets': {}, 'sources': []}
    for dataset, entries in loaded.items():
        seeds = sorted(m['train_seed'] for m, _ in entries)
        if set(seeds) - set(expected_seeds):
            raise ValueError('Unexpected training seed in ' + dataset)
        missing = sorted(set(expected_seeds) - set(seeds))
        if missing and not allow_partial:
            raise ValueError('Missing training seeds in {}: {}'.format(dataset, missing))
        reference_meta, reference_data = entries[0]
        def signature(meta):
            config = {k: v for k, v in meta['resolved_config'].items()
                      if k not in ('seed', 'stability_checkpoint_path', 'stability_fingerprints')}
            return stable_digest([meta['fingerprints'], config, meta['master_seed'], meta['num_masks'],
                                  meta['run_mode'], meta.get('runtime'), meta.get('sampling_code')])
        reference_signature = signature(reference_meta)
        for meta, data in entries:
            if signature(meta) != reference_signature:
                raise ValueError('Incompatible configuration/protocol/runtime in ' + dataset)
            for field in ('item_id', 'train_degree', 'group_code'):
                if not np.array_equal(data[field], reference_data[field]):
                    raise ValueError('Item alignment mismatch: ' + field)
            report['sources'].append({'dataset': dataset, 'seed': meta['train_seed'],
                                      'checkpoint_sha256': meta['checkpoint_sha256']})
        report['datasets'][dataset] = {'seeds': seeds, 'missing_seeds': missing,
                                      'seed_sd_available': len(seeds) >= 2,
                                      'run_mode': reference_meta['run_mode'],
                                      'num_masks': reference_meta['num_masks']}
        if missing:
            report['status'] = 'partial_exploratory'

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Summary output is not empty: ' + str(output))
    group_rows, fusion_group_rows, group_seed_rows = [], [], []
    with (output / 'item_mask_statistics.csv').open('w', newline='', encoding='utf-8-sig') as stream, \
         (output / 'fusion_statistics.csv').open('w', newline='', encoding='utf-8-sig') as fstream, \
         (output / 'item_seed_statistics.csv').open('w', newline='', encoding='utf-8-sig') as sstream:
        fields = ['dataset', 'train_seed', 'item_id', 'degree', 'group', 'modality']
        fields += [name + suffix for name in MEASURES for suffix in ('_mean', '_mask_sd')]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        fwriter = csv.DictWriter(fstream, fieldnames=['dataset', 'train_seed', 'item_id', 'degree', 'group',
                                  'activation_frequency', 'zero_clean', 'fusion_absolute_drift',
                                  'fusion_relative_drift', 'fusion_mean_norm', 'fusion_near_zero'])
        fwriter.writeheader()
        swriter = csv.DictWriter(sstream, fieldnames=['dataset', 'item_id', 'degree', 'group', 'modality', 'n_seeds'] +
                                                   [name + '_seed_sd' for name in MEASURES])
        swriter.writeheader()
        for dataset, entries in sorted(loaded.items()):
            seed_means = {name: [] for name in MEASURES}
            for meta, data in sorted(entries, key=lambda entry: entry[0]['train_seed']):
                stats = {name: mask_statistics(data[name]) for name in MEASURES}
                for name in MEASURES:
                    seed_means[name].append(stats[name][0])
                for index, item in enumerate(data['item_id']):
                    base = dict(dataset=dataset, train_seed=meta['train_seed'], item_id=int(item),
                                degree=int(data['train_degree'][index]), group=GROUP_LABELS[data['group_code'][index]])
                    for m, label in enumerate(('visual', 'text')):
                        row = dict(base, modality=label)
                        for name in MEASURES:
                            row[name + '_mean'] = stats[name][0][index, m]
                            row[name + '_mask_sd'] = stats[name][1][index, m]
                        writer.writerow(row)
                    frow = dict(base, activation_frequency=float((data['cosine_vt'][:, index] < 0).mean()),
                                zero_clean=bool(np.any(data['clean_norm'][:, index] <= 1e-8)))
                    for name in ('fusion_absolute_drift', 'fusion_relative_drift', 'fusion_mean_norm', 'fusion_near_zero'):
                        frow[name] = data[name][index]
                    fwriter.writerow(frow)
                for g, group in enumerate(GROUP_LABELS):
                    take = data['group_code'] == g
                    for m, modality in enumerate(('visual', 'text')):
                        for name in MEASURES:
                            group_rows.append(dict(dataset=dataset, train_seed=meta['train_seed'], group=group,
                                                   modality=modality, measure=name, **distribution(stats[name][1][take, m])))
                    fusion_group_rows.append(dict(dataset=dataset, train_seed=meta['train_seed'], group=group,
                        activation_frequency=float((data['cosine_vt'][:, take] < 0).mean()) if take.any() else None,
                        zero_clean_count=int(np.any(data['clean_norm'][:, take] <= 1e-8, axis=(0, 2)).sum()),
                        **distribution(data['fusion_absolute_drift'][take])))
            if len(entries) >= 2:
                data = entries[0][1]
                seed_sd = {name: seed_statistics(np.stack(values)) for name, values in seed_means.items()}
                for i, item in enumerate(data['item_id']):
                    for m, modality in enumerate(('visual', 'text')):
                        row = dict(dataset=dataset, item_id=int(item), degree=int(data['train_degree'][i]),
                                   group=GROUP_LABELS[data['group_code'][i]], modality=modality, n_seeds=len(entries))
                        row.update({name + '_seed_sd': seed_sd[name][i, m] for name in MEASURES})
                        swriter.writerow(row)
                for g, group in enumerate(GROUP_LABELS):
                    for m, modality in enumerate(('visual', 'text')):
                        for name in MEASURES:
                            group_seed_rows.append(dict(dataset=dataset, group=group, modality=modality,
                                measure=name, n_seeds=len(entries), **distribution(seed_sd[name][data['group_code'] == g, m])))

    def write_rows(name, rows, empty_fields):
        with (output / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_fields)
            writer.writeheader()
            writer.writerows(rows)
    write_rows('group_mask_statistics.csv', group_rows, [])
    write_rows('group_seed_statistics.csv', group_seed_rows, ['dataset', 'group', 'modality', 'measure', 'n_seeds', 'count', 'median', 'q25', 'q75', 'p90'])
    write_rows('fusion_group_statistics.csv', fusion_group_rows, [])
    pooled = defaultdict(list)
    for row in group_rows:
        if row['median'] is not None:
            pooled[(row['dataset'], row['group'], row['modality'], row['measure'])].append(row['median'])
    across = [dict(dataset=key[0], group=key[1], modality=key[2], measure=key[3], n_seeds=len(values),
                   mean_of_seed_medians=float(np.mean(values)),
                   sd_of_seed_medians=float(np.std(values, ddof=1)) if len(values) > 1 else None)
              for key, values in sorted(pooled.items())]
    write_rows('group_mask_across_seeds.csv', across, [])
    report['note'] = 'Cross-seed SD uses mask means and retains finite-mask Monte Carlo error. Single-seed SD is unavailable, not zero.'
    atomic_json(output / 'summary_manifest.json', report)
    return report
