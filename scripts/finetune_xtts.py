"""Launch Coqui's XTTS-v2 fine-tuning interface on localhost only.

The upstream demo currently requests a public Gradio share and binds to all
network interfaces. This wrapper patches that launch call in a temporary copy
and refuses to run if the upstream launch code has changed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-path", default="models/xtts-finetuned", help="Checkpoint and training output directory")
    parser.add_argument("--port", type=int, default=5003)
    args = parser.parse_args()

    try:
        import TTS
    except ImportError as exc:
        parser.error("Install project dependencies first: pip install -e '.[voice]'")
        raise exc

    upstream = Path(TTS.__file__).parent / "demos/xtts_ft_demo/xtts_demo.py"
    if not upstream.is_file():
        parser.error("This coqui-tts package does not include its XTTS fine-tuning demo")
    source = upstream.read_text(encoding="utf-8")
    unsafe_launch = 'demo.launch(share=True, debug=False, server_port=args.port, server_name="0.0.0.0")'
    safe_launch = 'demo.launch(share=False, debug=False, server_port=args.port, server_name="127.0.0.1")'
    if unsafe_launch not in source:
        parser.error("Upstream demo launch code changed; review it and update this local-only wrapper before continuing")

    with tempfile.TemporaryDirectory(prefix="local-xtts-finetune-") as tmp:
        patched_demo = Path(tmp) / "xtts_demo_local.py"
        patched_demo.write_text(source.replace(unsafe_launch, safe_launch), encoding="utf-8")
        command = [
            sys.executable,
            str(patched_demo),
            "--out_path",
            str(Path(args.out_path).expanduser().resolve()),
            "--port",
            str(args.port),
        ]
        print("Starting upstream XTTS fine-tuning UI on http://127.0.0.1:%s" % args.port)
        print("Voice recordings and checkpoints stay on this machine.")
        return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
