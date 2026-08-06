"""Show per-thread status."""

import sys
from pathlib import Path

pid = sys.argv[1] if len(sys.argv) > 1 else 1
base = Path(f"/proc/{pid}/task")
if not base.is_dir():
    print(f"no task dir for {pid}")
    raise SystemExit(0)
for entry in sorted(base.iterdir(), key=lambda p: int(p.name)):
    try:
        with (entry / "status").open(encoding="utf-8") as handle:
            status = handle.read()
        state_line = next((line for line in status.splitlines() if line.startswith("State:")), "?")
        name_line = next((line for line in status.splitlines() if line.startswith("Name:")), "?")
        syscall = next((line for line in status.splitlines() if line.startswith("Syscall:")), None)
        stack_lines = [line for line in status.splitlines() if line.startswith("Stack:")]
        print(f"tid={entry.name:>4}  {name_line}  {state_line}")
        if syscall:
            print(f"   {syscall}")
        for stack_line in stack_lines:
            print(f"   {stack_line}")
        # Check syscall file
        syscall_file = entry / "syscall"
        if syscall_file.is_file():
            try:
                value = syscall_file.read_text(encoding="utf-8").strip()
                print(f"   syscall_nr: {value}")
            except OSError:
                pass
    except OSError:
        continue
