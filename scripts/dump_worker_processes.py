"""Show full process tree inside the container."""

from pathlib import Path

base = Path("/proc")
entries = []
for entry in sorted(
    base.iterdir(),
    key=lambda item: int(item.name) if item.name.isdigit() else 0,
):
    if not entry.name.isdigit():
        continue
    pid = entry.name
    try:
        with (entry / "comm").open(encoding="utf-8") as handle:
            name = handle.read().strip()
        with (entry / "stat").open(encoding="utf-8") as handle:
            stat = handle.read().split()
        with (entry / "status").open(encoding="utf-8") as handle:
            status_lines = handle.read().splitlines()
        state = stat[2] if len(stat) > 2 else "?"
        ppid = stat[3] if len(stat) > 3 else "?"
        threads = stat[19] if len(stat) > 19 else "?"
        for line in status_lines:
            if line.startswith("Threads:"):
                threads = line.split(":", 1)[1].strip()
                break
        with (entry / "cmdline").open(encoding="utf-8") as handle:
            cmdline = handle.read().strip().replace("\x00", " ")
        entries.append(
            (
                int(pid),
                int(ppid) if ppid.isdigit() else -1,
                state,
                threads,
                name,
                cmdline,
            )
        )
    except OSError:
        continue

for pid, ppid, state, threads, name, cmdline in entries:
    cmd_preview = cmdline[:100]
    print(
        f"pid={pid:>5} ppid={ppid:>5} state={state} threads={threads:>3} "
        f"comm={name:>20} cmdline={cmd_preview}"
    )
