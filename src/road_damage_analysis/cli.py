from __future__ import annotations

import os
from pathlib import Path

import typer
import uvicorn

from road_damage_analysis.config import load_settings

app = typer.Typer(help="Global road damage monitoring pipeline.")


@app.command()
def collect_and_detect(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    download_limit: int = typer.Option(50, "--download-limit", min=1),
) -> None:
    """Collect street-view imagery and run damage detection."""
    from road_damage_analysis.pipeline import run_pipeline

    project_root = Path.cwd()
    settings = load_settings(config)
    outputs = run_pipeline(project_root, settings, download_limit=download_limit)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command()
def serve(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Serve the dashboard."""
    from road_damage_analysis.webapp import create_app

    project_root = Path.cwd()
    settings = load_settings(config)
    app_instance = create_app(project_root, settings)
    host = os.getenv('HOST', settings.web.host)
    port = int(os.getenv('PORT', str(settings.web.port)))
    uvicorn.run(app_instance, host=host, port=port, reload=reload)
