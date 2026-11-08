"""Command line: `python -m app.cli submit CONFIG.yaml [--wait]` and `python -m app.cli seed`."""

import argparse
import sys
from pathlib import Path

import yaml

from app.db.repositories import Job, JobRepository
from app.db.session import get_engine
from app.jobs import service
from app.jobs.config import ConfigError


def _describe(job: Job) -> str:
    line = f"job {job.id} ({job.name}): {job.status}"
    if job.mlflow_run_id:
        line += f", MLflow run {job.mlflow_run_id}"
    if job.error:
        line += f"\n  error: {job.error}"
    return line


def _submit(args: argparse.Namespace) -> int:
    try:
        raw = yaml.safe_load(Path(args.config).read_text())
    except (OSError, yaml.YAMLError) as exc:
        print(f"cannot read {args.config}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(raw, dict):
        print(f"{args.config}: expected a YAML mapping", file=sys.stderr)
        return 2

    jobs = JobRepository(get_engine())
    try:
        job = service.submit(jobs, raw)
    except ConfigError as exc:
        print(f"{args.config}: {exc}", file=sys.stderr)
        return 2
    print(f"queued {_describe(job)}; {job.progress_total} Candidate(s) to evaluate")
    if not args.wait:
        return 0

    print("waiting for the worker (is the stack running? `make up`)…")
    job = service.wait(jobs, job.id)
    print(_describe(job))
    return 0 if job.status == "completed" else 1


def _seed(args: argparse.Namespace) -> int:
    # Imported here, so `submit` does not pay for the registry and HTTP clients.
    from app.seed import SeedError, seed

    try:
        seed()
    except SeedError as exc:
        print(f"seed failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="validate an Experiment Config and queue a job")
    submit.add_argument("config", help="path to an Experiment Config YAML file")
    submit.add_argument("--wait", action="store_true", help="block until the job finishes")
    submit.set_defaults(handler=_submit)
    seed = commands.add_parser(
        "seed", help="train, register and deploy one Run per seed config (needs `make up`)"
    )
    seed.set_defaults(handler=_seed)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
