from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@dataclass(frozen=True)
class RunContext:
    """Convenience wrapper around per-run output paths."""

    base_dir: Path
    date_stamp: str
    run_name: str

    @property
    def run_dir(self) -> Path:
        return self.base_dir / self.date_stamp / self.run_name

    def ensure(self) -> Path:
        path = self.run_dir
        path.mkdir(parents=True, exist_ok=False)
        return path


def make_unique_run_name(base_dir: Path, date_stamp: str, run_name: str) -> str:
    base_path = base_dir / date_stamp / run_name
    if not base_path.exists():
        return run_name

    i = 1
    while True:
        candidate = f"{run_name}_{i}"
        if not (base_dir / date_stamp / candidate).exists():
            return candidate
        i += 1

def build_run_context(cfg: DictConfig) -> RunContext:
    base = Path(to_absolute_path(cfg.output_directory))
    date_stamp = datetime.today().strftime("%Y-%m-%d")
    raw_name = cfg.model.wandb_name

    run_name = make_unique_run_name(base, date_stamp, raw_name)

    return RunContext(base_dir=base, date_stamp=date_stamp, run_name=run_name)
