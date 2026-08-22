import sys
import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Loads YAML configuration from an explicit path, CLI argument (--config <path>),
    or standard fallback locations.
    """
    # 1. Direct argument
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)

    # 2. CLI argument: --config <path>
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            cli_path = Path(sys.argv[idx + 1])
            if cli_path.exists():
                with open(cli_path, "r") as f:
                    return yaml.safe_load(f)

    # 3. Fallback candidates
    candidates = [
        Path("/app/config/config-docker.yml"),
        Path(__file__).resolve().parent.parent.parent / "config" / "config-local.yml",
        Path("config/config-local.yml"),
        Path("config/config-docker.yml"),
    ]

    for p in candidates:
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(f"Configuration file not found. Checked: {config_path or 'default candidates'}")
