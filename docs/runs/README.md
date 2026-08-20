# Collected run logs

Logs from runs on the cluster, stripped of tqdm's carriage-return rewrites and
committed so a result can be read in a diff instead of a terminal paste.

Collected with, from the repository root:

    bash scripts/collect_runs.sh docs/runs/<experiment> ~/<pattern>*.log

Each run contributes two files: `<name>.log` (the hyper-parameter header, one
line per epoch, the test block) and `<name>.json` (`test_results.json` copied
from the run directory, which is the authoritative result -- the log is a
transcript and can be truncated by a job hitting its wall clock).

Checkpoints are not collected. They are large, `*.pth` is in `.gitignore`, and
nothing about a result needs them.
