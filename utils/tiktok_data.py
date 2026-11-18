"""Read-only source audit and lossless split conversion for the local TikTok files."""
import csv
import hashlib
import io
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from utils.atomic_io import atomic_bytes, atomic_json, exclusive_lock, file_hash, read_json

MATRICES = [('train', 'trnMat.pkl'), ('valid', 'valMat.pkl'), ('test', 'tstMat.pkl')]
FEATURES = ['image_feat.npy', 'text_feat.npy', 'audio_feat.npy']


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('scipy.sparse.coo', 'coo_matrix'): sp.coo_matrix,
                   ('scipy.sparse._coo', 'coo_matrix'): sp.coo_matrix,
                   ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
                   ('numpy._core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
                   ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError('Unexpected pickle class: ' + module + '.' + name)
        return allowed[(module, name)]


def inspect_source(source):
    source = Path(source).resolve()
    names = [name for _, name in MATRICES] + FEATURES
    hashes = {name: file_hash(source / name) for name in names}
    matrices, shape = {}, None
    for split, name in MATRICES:
        with (source / name).open('rb') as stream:
            matrix = DataOnlyUnpickler(stream).load()
        if not sp.issparse(matrix):
            raise ValueError('Expected sparse matrix: ' + name)
        matrix = matrix.tocoo()
        if len(matrix.shape) != 2 or min(matrix.shape) <= 0:
            raise ValueError('Invalid matrix shape')
        if shape is not None and matrix.shape != shape:
            raise ValueError('Inconsistent matrix shapes')
        shape = matrix.shape
        if not matrix.nnz or not np.isfinite(matrix.data).all() or not (matrix.data == 1).all():
            raise ValueError('Expected nonempty finite binary interactions: ' + name)
        if ((matrix.row < 0).any() or (matrix.row >= shape[0]).any() or
                (matrix.col < 0).any() or (matrix.col >= shape[1]).any()):
            raise ValueError('Invalid interaction index')
        keys = matrix.row.astype(np.int64) * shape[1] + matrix.col
        if np.unique(keys).size != matrix.nnz:
            raise ValueError('duplicate interactions: ' + name)
        matrices[split] = matrix
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        if matrices[a].tocsr().multiply(matrices[b].tocsr()).nnz:
            raise ValueError('Split overlap: ' + a + '/' + b)
    features = {}
    for name in FEATURES:
        a = np.load(source / name, mmap_mode='r', allow_pickle=False)
        if a.ndim != 2 or a.shape[0] != shape[1] or a.shape[1] < 1:
            raise ValueError('Feature rows must match item space: ' + name)
        if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
            raise ValueError('Feature must be numeric and finite: ' + name)
        features[name] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                          'zero_rows': int(np.all(a == 0, axis=1).sum()),
                          'used': name != 'audio_feat.npy'}
    train = matrices['train']
    users = np.unique(train.row)
    counts = {k: int(v.nnz) for k, v in matrices.items()}
    effective = {k: int(np.isin(v.row, users).sum()) for k, v in matrices.items()}
    report = {'format_version': 1, 'source': str(source), 'source_hashes': hashes,
              'provenance': 'UNVERIFIED: confirm dataset source, indexing and evaluation protocol',
              'n_users': shape[0], 'n_items': shape[1], 'counts': counts,
              'effective_counts': effective, 'features': features,
              'train_zero_users': int(shape[0] - len(users)),
              'train_zero_items': int(shape[1] - np.unique(train.col).size),
              'sparsity_percent': 100 * (1 - sum(counts.values()) / (shape[0] * shape[1])),
              'protocol': {'preserve_source_splits': True, 'filter_train_zero_users': True,
                           'retain_all_item_candidates': True, 'audio_used': False,
                           'evaluation_mask': 'training interactions only, for validation and test'}}
    return matrices, report


def validate_prepared(output):
    output = Path(output).resolve()
    report = read_json(output / 'audit.json')
    for name, expected in report['source_hashes'].items():
        if file_hash(Path(report['source']) / name) != expected:
            raise ValueError('Source changed: ' + name)
    if file_hash(output / 'tiktok.inter') != report['interaction_sha256']:
        raise ValueError('Prepared interactions changed')
    return report


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('Prepared output must be outside original data directory')
    with exclusive_lock(output.parent / (output.name + '.prepare.lock')):
        if (output / 'audit.json').exists():
            report = validate_prepared(output)
            if Path(report['source']) != source:
                raise ValueError('Source changed; choose a new output directory')
            return report
        if (output / 'tiktok.inter').exists():
            raise ValueError('Incomplete conversion; choose a new output directory')
        matrices, report = inspect_source(source)
        text = io.StringIO(newline='')
        writer = csv.writer(text, delimiter='\t', lineterminator='\n')
        writer.writerow(['userID', 'itemID', 'x_label'])
        for label, (split, _) in enumerate(MATRICES):
            matrix = matrices[split]
            for i in np.lexsort((matrix.col, matrix.row)):
                writer.writerow([int(matrix.row[i]), int(matrix.col[i]), label])
        data = text.getvalue().encode('utf-8')
        report['interaction_sha256'] = hashlib.sha256(data).hexdigest()
        atomic_bytes(output / 'tiktok.inter', data)
        atomic_json(output / 'audit.json', report)
        validate_prepared(output)
        return report
