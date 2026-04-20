"""Typer CLI entry — `verbum <subcommand>`.

Per AGENTS.md S1 λ interface: CLI is the secondary surface (batch,
automation, CI). Logic lives in the library; the CLI wraps it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from verbum import __version__

app = typer.Typer(
    name="verbum",
    no_args_is_help=True,
    help="verbum — distilling the lambda compiler from LLMs.",
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Root callback — forces typer to dispatch on subcommands even when
    only one is registered. Keeps `verbum <subcommand>` the stable shape
    as commands are added.
    """


@app.command()
def version() -> None:
    """Print the installed verbum version."""
    typer.echo(__version__)


@app.command()
def run(
    probe_set: Annotated[
        Path,
        typer.Argument(help="Path to the probe-set JSON file."),
    ],
    gates_dir: Annotated[
        Path,
        typer.Option("--gates", help="Directory containing gate .txt files."),
    ] = Path("gates/"),
    results_dir: Annotated[
        Path,
        typer.Option("--results", help="Parent directory for result output."),
    ] = Path("results/"),
    server_url: Annotated[
        str | None,
        typer.Option("--server", help="llama.cpp server URL."),
    ] = None,
    n_predict: Annotated[
        int,
        typer.Option("--n-predict", help="Max tokens to generate per probe."),
    ] = 512,
    temperature: Annotated[
        float,
        typer.Option("--temperature", help="Sampling temperature."),
    ] = 0.0,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility."),
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option("--model", help="Model name for provenance."),
    ] = None,
) -> None:
    """Fire a probe set against the llama.cpp server and record results."""
    from verbum.runner import run_probe_set

    summary = run_probe_set(
        probe_set_path=probe_set,
        gates_dir=gates_dir,
        results_dir=results_dir,
        server_url=server_url,
        n_predict=n_predict,
        temperature=temperature,
        seed=seed,
        model_name=model_name,
        project_root=Path("."),
    )

    typer.echo("")
    typer.echo(f"Run:       {summary.run_id}")
    typer.echo(f"Results:   {summary.run_dir}")
    typer.echo(f"Total:     {summary.total}")
    typer.echo(f"Succeeded: {summary.succeeded}")
    typer.echo(f"Failed:    {summary.failed}")
    typer.echo(f"Elapsed:   {summary.elapsed_s:.1f}s")


if __name__ == "__main__":
    app()
