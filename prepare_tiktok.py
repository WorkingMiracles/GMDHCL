"""Convert TikTok's provided sparse splits without resplitting; use visual/text features only."""
import argparse
from pathlib import Path
import shutil

from utils.tiktok_data import prepare, validate_prepared
from utils.atomic_io import file_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Original directory with trnMat.pkl, valMat.pkl, tstMat.pkl and feature arrays')
    parser.add_argument('--output', type=Path, required=True,
                        help='A separate prepared directory named tiktok')
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if output.name != 'tiktok':
        raise ValueError('The prepared dataset directory must be named tiktok')
    report = prepare(source, output)
    for name in ('image_feat.npy', 'text_feat.npy'):
        destination = output / name
        if destination.exists() and file_hash(destination) != report['source_hashes'][name]:
            raise FileExistsError('Refusing to overwrite different prepared features: ' + str(destination))
        if not destination.exists():
            shutil.copy2(str(source / name), str(destination))
        if file_hash(destination) != report['source_hashes'][name]:
            raise ValueError('Feature copy checksum mismatch: ' + name)
    validate_prepared(output)
    print('Prepared:', output)
    print('Raw split counts:', report['counts'])
    print('Counts after excluding training-cold users:', report['effective_counts'])
    print('Audio is audited for provenance and is not an input to GMDHCL.')


if __name__ == '__main__':
    main()
