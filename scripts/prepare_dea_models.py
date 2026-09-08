"""Download immutable model snapshots; no medical data leaves this machine."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dea_v1.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    states = {}
    for key, spec in cfg["backbones"].items():
        states[key] = dict(status="downloading", **spec)
        (output / "model_status.json").write_text(json.dumps(states, indent=2))
        try:
            path = snapshot_download(spec["repo"], revision=spec["revision"],
                                     local_dir=ROOT / "models/pretrained" / key,
                                     allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.jinja"], max_workers=3)
            states[key].update(status="downloaded", path=path)
        except Exception as e:
            states[key].update(status="failed", error=repr(e))
        (output / "model_status.json").write_text(json.dumps(states, indent=2))
        print(json.dumps(states[key]), flush=True)


if __name__ == "__main__":
    main()
