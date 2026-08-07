"""Resolve manifest image paths against a dataset root.

Manifests store each image as a path *relative to the dataset root* so that a
prepared profile is portable: the same 175 MB of partition artifacts describe
the same experiment whether the images live under ``F:\\project\\...\\data\\raw``
on a Windows laptop, ``/app/data/raw`` in a container, or ``/content/data/raw``
on a rented GPU box.  Only the root differs, and only the root is supplied at
run time.

Absolute paths are still accepted when reading.  Profiles prepared before this
change hold machine-specific absolute paths, and refusing them would strand
partitions that cost hours to build; they simply are not portable.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Fallback for the dataset root when no explicit path is passed. Useful on a
#: rented GPU machine where the run command comes from a saved script.
DATASET_ROOT_ENVIRONMENT_VARIABLE = "CROPFED_DATASET_ROOT"


def resolve_dataset_root(dataset_root: Path | str | None) -> Path | None:
    """Return the dataset root to resolve against, or ``None`` if unset.

    An explicit value always wins over the environment, so a script that passes
    ``--dataset-root`` cannot be silently redirected by a stale shell variable.
    """

    value = dataset_root if dataset_root is not None else os.getenv(
        DATASET_ROOT_ENVIRONMENT_VARIABLE
    )
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser().resolve()


def resolve_image_path(stored: str, dataset_root: Path | None) -> Path:
    """Turn a manifest ``path`` value into a path that can be opened.

    Raises rather than guessing when a relative path arrives without a root:
    silently resolving against the current working directory would make a run
    depend on where it was launched from, and the failure would surface as
    thousands of missing-file errors deep inside the audit.
    """

    path = Path(stored).expanduser()
    if path.is_absolute():
        return path
    if dataset_root is None:
        raise ValueError(
            f"manifest holds the relative image path {stored!r} but no dataset "
            "root was given; pass --dataset-root or set "
            f"{DATASET_ROOT_ENVIRONMENT_VARIABLE}"
        )
    return dataset_root / path


def to_manifest_path(image_path: Path, dataset_root: Path) -> str:
    """Return the portable manifest value for an image on disk.

    POSIX separators are used on every platform so a manifest written on
    Windows reads unchanged on Linux.
    """

    resolved = image_path.expanduser().resolve()
    root = dataset_root.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"image {resolved} is not inside the dataset root {root}"
        ) from error
