"""Basic testing datasets"""

from pathlib import Path

import numpy as np
import pytest

from zea.config import Config, check_config
from zea.data.datasets import Dataset, Folder, split_files_by_directory
from zea.internal.checks import _IMAGE_DATA_TYPES, _NON_IMAGE_DATA_TYPES

from .. import DUMMY_DATASET_GRID_SIZE_X, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_N_FRAMES

_ALL_DATA_TYPES = _IMAGE_DATA_TYPES + _NON_IMAGE_DATA_TYPES


@pytest.mark.parametrize(
    "file_idx, idx, expected_shape",
    [
        (
            0,
            "all",
            (DUMMY_DATASET_N_FRAMES, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            -1,
            (1, 2, 3),
            (),
        ),
        (
            0,
            [1, 2, 3],
            (3, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            -1,
            np.array([1, 2, 3]),
            (3, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            0,
            slice(1, 3),
            (2, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            -1,
            (0, list(range(5))),
            (5, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            0,
            (np.array([1, 2]), slice(10)),
            (2, 10, DUMMY_DATASET_GRID_SIZE_X),
        ),
        (
            0,
            (slice(None), np.arange(10)),
            (DUMMY_DATASET_N_FRAMES, 10, DUMMY_DATASET_GRID_SIZE_X),
        ),
    ],
)
def test_dataset_indexing(file_idx, idx, expected_shape, dummy_dataset_path):
    """Test ui initialization function"""
    config = {"data": {"dataset_folder": dummy_dataset_path, "dtype": "image"}}
    config = check_config(Config(config))
    dataset = Dataset.from_config(**config.data)

    file = dataset[file_idx]
    data = file.load_data(config.data.dtype, idx)

    assert data.shape == expected_shape, (
        f"Data shape {data.shape} does not match expected shape {expected_shape}"
    )


def test_folder_copy_key_by_key(dummy_dataset_path, tmp_path):
    """Test copying a `zea.Folder` key by key."""

    _copied_keys = []

    for key in _ALL_DATA_TYPES:
        _copied_keys.append(key)
        _other_keys = [k for k in _ALL_DATA_TYPES if k not in _copied_keys]

        # Copy the folder with the specified key
        folder = Folder(dummy_dataset_path, validate=False)
        folder.copy(tmp_path / "copy", key)

        # Check required keys in the copied folder
        with Dataset(tmp_path / "copy", validate=False) as copied_folder:
            for file in iter(copied_folder):
                for key in _copied_keys:
                    assert key in file["data"], f"Copied folder does not contain {key} key"
                assert "scan" in file, "Copied folder does not contain 'scan' key"

            # Check that the copied folder does not contain other keys
            for file in iter(copied_folder):
                for other_key in _other_keys:
                    assert other_key not in file["data"], (
                        f"Copied folder should not contain {other_key} key"
                    )


def test_folder_copy_all_keys(dummy_dataset_path, tmp_path):
    """Test copying a `zea.Folder` with all keys."""

    # Copy the folder
    folder = Folder(dummy_dataset_path, validate=False)
    folder.copy(tmp_path / "copy", key="all")

    # Check required keys in the copied folder
    with Dataset(tmp_path / "copy", validate=False) as copied_folder:
        for file in iter(copied_folder):
            for key in _ALL_DATA_TYPES:
                assert key in file["data"], f"Copied folder does not contain {key} key"
            assert "scan" in file, "Copied folder does not contain 'scan' key"


@pytest.mark.parametrize(
    "dir_sizes, splits, expected_counts",
    [
        # full split returns all files from both directories
        ([10, 20], [1.0, 1.0], [10, 20]),
        # half split from each directory
        ([10, 20], [0.5, 0.5], [5, 10]),
        # zero split from first directory, all from second
        ([10, 20], [0.0, 1.0], [0, 20]),
        # all from first, none from second
        ([10, 20], [1.0, 0.0], [10, 0]),
        # three directories, full split
        ([5, 5, 5], [1.0, 1.0, 1.0], [5, 5, 5]),
        # three directories, partial split (int truncation: int(0.6*5)=3)
        ([5, 5, 5], [0.6, 0.6, 0.6], [3, 3, 3]),
        # single directory
        ([8], [0.25], [2]),
    ],
)
def test_split_files_by_directory(dir_sizes, splits, expected_counts, tmp_path):
    """Test that split_files_by_directory returns the correct number of files per directory."""

    # Build fake file paths (no real files needed)
    directories = [str(tmp_path / f"dir{i}") for i in range(len(dir_sizes))]
    file_names = []
    for dir_path, n_files in zip(directories, dir_sizes):
        for j in range(n_files):
            file_names.append(str(Path(dir_path) / f"file{j:04d}.hdf5"))

    result = split_files_by_directory(file_names, directories, splits)

    assert len(result) == sum(expected_counts), (
        f"Expected {sum(expected_counts)} files, got {len(result)}"
    )

    # Verify the correct number of files was taken from each directory
    for dir_path, expected in zip(directories, expected_counts):
        count = sum(1 for f in result if f.startswith(dir_path))
        assert count == expected, f"Expected {expected} files from '{dir_path}', got {count}"
