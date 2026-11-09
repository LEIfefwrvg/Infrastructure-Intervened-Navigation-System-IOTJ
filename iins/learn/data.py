from pathlib import Path
import shutil

import numpy as np


DATA_ROOT = Path(__file__).resolve().parent / 'weights'
FIELDS = ('occupancy', 'trajectories', 'agent_mask', 'flow', 'flow_mask',
          'centers', 'feasible')


def load(path):
    """Load input tensors and labels in the training/evaluation field order."""
    with np.load(path, allow_pickle=False) as data:
        return [data[key] for key in FIELDS]


def generate(path, *, split='train', dataset='etc_derived'):
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    source = DATA_ROOT / f'{split}.npz'
    if not source.is_file():
        raise FileNotFoundError(f'Bundled dataset missing: {source}')
    destination = Path(path)
    if destination.resolve() == source.resolve():
        return destination
    if destination.exists():
        if destination.read_bytes() == source.read_bytes():
            return destination
        raise FileExistsError(f'Refusing to overwrite different data: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination
