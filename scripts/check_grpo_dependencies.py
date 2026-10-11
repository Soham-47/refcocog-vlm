#!/usr/bin/env python3
"""Fail fast when the pinned GRPO-only dependencies are unavailable."""

import importlib
import sys
from importlib import metadata
from pathlib import Path


MODULES = {
    "msgspec": "msgspec",
    "eval-type-backport": "eval_type_backport",
}


def read_requirements(path):
    requirements = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version or name not in MODULES:
            raise ValueError(f"Unsupported GRPO requirement on line {line_number}: {raw_line}")
        requirements[name] = version
    return requirements


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(f"Usage: {Path(sys.argv[0]).name} REQUIREMENTS", file=sys.stderr)
        return 2

    requirements_path = Path(argv[0])
    try:
        requirements = read_requirements(requirements_path)
    except (OSError, ValueError) as error:
        print(f"GRPO dependency preflight failed: {error}", file=sys.stderr)
        return 1

    failures = []
    for distribution, expected_version in requirements.items():
        try:
            installed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            failures.append(f"{distribution}=={expected_version} is missing")
            continue

        if installed_version != expected_version:
            failures.append(
                f"{distribution} requires {expected_version}, found {installed_version}"
            )
            continue

        try:
            importlib.import_module(MODULES[distribution])
        except ImportError as error:
            failures.append(f"{distribution}=={expected_version} cannot be imported: {error}")

    if failures:
        print("GRPO dependency preflight failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        print(
            f"Install once in this environment with: {sys.executable} -m pip install -r {requirements_path}",
            file=sys.stderr,
        )
        return 1

    print(
        "GRPO dependency preflight passed: "
        + ", ".join(f"{name}=={version}" for name, version in requirements.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
