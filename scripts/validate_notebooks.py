"""Validate the six supplied notebooks and optionally replay archived CPU results."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]


def results_start(book, cell_id):
    matches = [i for i, cell in enumerate(book.cells) if cell.id == cell_id]
    if len(matches) != 1:
        raise ValueError("The recorded CPU analysis start cell is missing or duplicated")
    if book.cells[matches[0]].cell_type != "code":
        raise ValueError("The CPU analysis must start at a code cell")
    return matches[0]


def validate():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())["notebooks"]
    paths = [ROOT / entry["repository_path"] for entry in manifest]
    assert len(paths) == len(set(paths)) == 6
    assert set(paths) == set((ROOT / "notebooks").glob("*.ipynb"))
    transform = TransformerManager()
    for entry, path in zip(manifest, paths):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["repository_sha256"], path.name
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        assert len(book.cells) == entry["cell_count"]
        results_start(book, entry["results_start_cell_id"])
        for i, cell in enumerate(book.cells):
            if cell.cell_type == "code":
                ast.parse(transform.transform_cell(cell.source), filename=f"{path.name}:cell-{i}")
                assert not any(output.output_type == "error" for output in cell.outputs), (path.name, i)
        print(f"Validated {path.name}: {len(book.cells)} cells", flush=True)
    return paths


def report_worker(path):
    """A fresh process runs only the trusted notebook's historical reporting section."""
    book = nbformat.read(path, as_version=4)
    entries = json.loads((ROOT / "source_manifest.json").read_text())["notebooks"]
    relative_path = path.resolve().relative_to(ROOT).as_posix()
    entry = next(item for item in entries if item["repository_path"] == relative_path)
    start = results_start(book, entry["results_start_cell_id"])
    namespace = {"__name__": "__main__"}
    for i, cell in enumerate(book.cells[start:], start=start):
        if cell.cell_type == "code":
            print(f"Running {path.name}:cell-{i}", flush=True)
            exec(compile(cell.source, f"{path.name}:cell-{i}", "exec"), namespace)
    # Report mode must not accidentally enter the GPU setup or use a live-run folder.
    assert "model" not in namespace and "OUT" not in namespace
    figures = list(Path.cwd().rglob("*.png"))
    assert figures and all(path.stat().st_size > 0 for path in figures)
    print(f"CPU replay complete: {len(figures)} figures", flush=True)


def replay(paths, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        work = output / path.stem
        work.mkdir(exist_ok=True)
        environment = dict(os.environ, MPLBACKEND="Agg",
                           MPLCONFIGDIR=str(work / ".matplotlib"),
                           IPYTHONDIR=str(work / ".ipython"))
        log = work / "replay.log"
        with log.open("w") as stream:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--report-worker", str(path)],
                cwd=work, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError(f"CPU replay failed for {path.name}. See {log}")
        print(f"CPU replay passed: {path.name}", flush=True)
    print(f"CPU replay outputs: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Recompute archived Results sections on CPU")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cpu-replay")
    parser.add_argument("--report-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.report_worker:
        report_worker(args.report_worker)
        return
    paths = validate()
    if args.replay:
        replay(paths, args.output)


if __name__ == "__main__":
    main()
