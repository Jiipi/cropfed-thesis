"""Look at one manifest row to see the path that is failing."""

import csv
import sys
from pathlib import Path

target = Path(sys.argv[1])
with target.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for i, row in enumerate(reader):
        if i > 5:
            break
        path_value = row["path"]
        path_obj = Path(path_value)
        exists = path_obj.is_file()
        print(f"row {i}: exists={exists} path={path_value}")
        if exists:
            print(f"  size={path_obj.stat().st_size}")
        else:
            # Try resolving through parent
            parent = path_obj.parent
            print(f"  parent exists: {parent.is_dir()}")
            if parent.is_dir():
                siblings = sorted(parent.iterdir())[:3]
                for sib in siblings:
                    print(f"    sibling: {sib.name}")