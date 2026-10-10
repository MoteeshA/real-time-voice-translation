"""Validate and train the public GLOBE XTTS pilot on a CUDA machine.

No model is loaded or downloaded unless the data and hardware preflights pass.
--allow-cpu is for diagnostics on sufficiently large machines, not this 8 GB Mac.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, distributions, version
import json
from pathlib import Path
import platform
import shutil
import sys
import uuid

FIELDS = ("audio_file", "text", "speaker_name")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dataset(path: Path) -> list[dict[str, str]]:
    """Resolve audio relative to its CSV, matching the Coqui formatter."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected pipe-delimited columns {FIELDS}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    seen = set()
    for number, row in enumerate(rows, 2):
        if None in row or any(not isinstance(row.get(field), str) or not row[field].strip() for field in FIELDS):
            raise ValueError(f"{path}:{number}: malformed or empty field")
        audio = (path.parent / row["audio_file"]).resolve()
        if not audio.is_file():
            raise ValueError(f"{path}:{number}: missing audio {audio}")
        if audio in seen:
            raise ValueError(f"{path}:{number}: duplicate audio {audio}")
        seen.add(audio)
        row["audio_file"] = str(audio)
    return rows


def validate_dataset(train_csv: Path, eval_csv: Path) -> tuple[dict, dict]:
    splits = {"train": read_dataset(train_csv), "eval": read_dataset(eval_csv)}
    speakers = {key: {row["speaker_name"] for row in rows} for key, rows in splits.items()}
    overlap = speakers["train"] & speakers["eval"]
    if overlap:
        raise ValueError(f"Training/evaluation speakers overlap: {sorted(overlap)}")
    audio_overlap = {r["audio_file"] for r in splits["train"]} & {r["audio_file"] for r in splits["eval"]}
    if audio_overlap:
        raise ValueError("Training and evaluation share an audio file")
    report = {key: {"clips": len(rows), "speakers": sorted(speakers[key])} for key, rows in splits.items()}
    report["speaker_overlap"] = []
    report["csv_sha256"] = {"train": sha256_file(train_csv), "eval": sha256_file(eval_csv)}
    return splits, report


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_finite(values: dict, context: str) -> None:
    """Reject invalid losses before backward, and invalid weights before export."""
    import torch

    for name, value in values.items():
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"Non-finite {context}: {name}")


def correct_prompt_indices(indices, code_stride: int, prompt_compression: int):
    """Compensate for Coqui 0.27.5 converting audio masks on the prompt grid.

    GPT.forward divides these waveform positions by prompt_compression, but
    applies them to DVAE targets whose stride is code_stride. Preserve that
    upstream division while ensuring its resulting indices use the DVAE grid.
    """
    return indices * prompt_compression // code_stride


def check_audio_decoder(splits: dict) -> None:
    """Surface a broken codec before XTTS recursively retries every sample."""
    import torch
    import torchaudio
    for split, rows in splits.items():
        path = rows[0]["audio_file"]
        try:
            audio, rate = torchaudio.load(path)
            if rate <= 0 or not audio.numel() or not torch.isfinite(audio).all().item():
                raise ValueError("Empty or invalid decoded audio")
        except Exception as exc:
            raise RuntimeError(f"Public {split} audio decode preflight failed: {path}: {exc}") from exc


def export_model_snapshot(model_state: dict, destination: Path, step: int, epoch: int) -> dict:
    """Convert live weights directly, avoiding a second full checkpoint mapping."""
    import torch
    ignored = {"torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"}
    state = {name: tensor.to(device="cpu", dtype=torch.float16) if tensor.is_floating_point() else tensor.cpu()
             for name, tensor in model_state.items()
             if name.removeprefix("xtts.").split(".")[0] not in ignored}
    require_finite(state, "compact checkpoint parameter")
    torch.save({"model": state, "step": step, "epoch": epoch}, destination)
    return {"step": step, "epoch": epoch, "optimizer_saved": False,
            "floating_dtype": "float16", "sha256": sha256_file(destination)}


def export_weight_snapshot(checkpoint: Path, destination: Path) -> dict:
    """Save compact warm-start weights without multi-GB optimizer state."""
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    return export_model_snapshot(saved["model"], destination, saved["step"], saved["epoch"])


def enable_memory_saving(transformer) -> None:
    transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    transformer.config.use_cache = False


def prepare_training_dataset(splits: dict, destination: Path) -> None:
    """Stage audio beneath the CSV root required by Coqui's sample identifiers."""
    destination.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        audio_dir = destination / "audio" / split
        audio_dir.mkdir(parents=True)
        with (destination / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="|")
            writer.writeheader()
            for index, row in enumerate(rows):
                source = Path(row["audio_file"])
                target = audio_dir / f"{index:06d}{source.suffix}"
                shutil.copyfile(source, target)
                writer.writerow({**{key: row[key] for key in FIELDS},
                                 "audio_file": target.relative_to(destination).as_posix()})


def export_inference(trainer_path: Path, original_config: Path, vocab: Path, destination: Path) -> dict:
    """Keep a loadable XTTS bundle; the training config is not an inference config."""
    import torch

    checkpoint = trainer_path / "best_model.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"No best_model.pth after training: {trainer_path}")
    # This is our own just-generated trainer checkpoint. mmap avoids eagerly
    # allocating all optimizer tensors while producing a smaller inference file.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    ignored = {"torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"}
    state = {}
    for name, tensor in saved["model"].items():
        name = name.removeprefix("xtts.")
        if name.split(".")[0] not in ignored:
            state[name] = tensor
    if not state:
        raise RuntimeError("Training checkpoint contains no inference parameters")
    require_finite(state, "inference parameter")
    destination.mkdir(parents=True, exist_ok=False)
    torch.save({"model": state}, destination / "model.pth")
    for source, name in ((original_config, "config.json"), (vocab, "vocab.json")):
        shutil.copyfile(source, destination / name)
    return {"directory": str(destination), "source_checkpoint": str(checkpoint),
            "optimizer_removed": True, "training_only_parameters_removed": True,
            "sha256": {path.name: sha256_file(path) for path in sorted(destination.iterdir())}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/globe-v2-india-south-asia/pilot/train.csv"))
    parser.add_argument("--eval-csv", type=Path, default=Path("data/globe-v2-india-south-asia/pilot/eval.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/xtts-india-south-asia-pilot"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--print-step", type=int, default=10, help="Log every N training batches")
    parser.add_argument("--save-step", type=int, default=50, help="Checkpoint every N training batches")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--backup-dir", type=Path, help="Persistent storage for the latest complete checkpoint and logs")
    parser.add_argument("--initial-checkpoint", type=Path, help="Warm-start model weights from a prior trainer checkpoint; optimizer resets")
    parser.add_argument("--download-snapshots", action="store_true", help="Announce compact local weight snapshots for notebook downloads")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--check-data-only", action="store_true", help="Validate CSVs; exit without importing torch")
    parser.add_argument("--allow-cpu", action="store_true", help="Explicit diagnostic override; very memory intensive")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.grad_accum, args.print_step, args.save_step, args.torch_threads) < 1:
        parser.error("epochs, batch size, accumulation, print step, save step, and torch threads must be positive")
    if args.workers < 0 or not 0 <= args.seed < 2**32:
        parser.error("workers must be non-negative; seed must be between 0 and 2^32-1")
    train_csv, eval_csv = args.train_csv.expanduser().resolve(), args.eval_csv.expanduser().resolve()
    try:
        splits, dataset_report = validate_dataset(train_csv, eval_csv)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(dataset_report, indent=2), flush=True)
    if args.check_data_only:
        return 0
    try:
        import torch
    except ImportError:
        parser.error("Install a CUDA-enabled PyTorch build and the training requirements first")
    if not torch.cuda.is_available() and not args.allow_cpu:
        parser.error("CUDA is unavailable. No model was loaded. Train on a CUDA GPU; --allow-cpu is diagnostic only.")
    torch.set_num_threads(args.torch_threads)
    try:
        check_audio_decoder(splits)
    except RuntimeError as exc:
        parser.error(str(exc))
    try:
        from TTS.demos.xtts_ft_demo.utils import gpt_train
    except ImportError:
        parser.error("Install coqui-tts[codec]==0.27.5 and the training requirements first")
    if version("coqui-tts") != "0.27.5":
        parser.error("This launcher targets coqui-tts==0.27.5; review its trainer API before using another version")
    if version("coqui-tts-trainer") != "0.3.3":
        parser.error("This launcher targets coqui-tts-trainer==0.3.3; install the pinned training requirements")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_output = output / run_id
    run_output.mkdir()
    metadata_path = run_output / "run_metadata.json"
    backup = args.backup_dir.expanduser().resolve() / run_id if args.backup_dir else None
    if backup:
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(Path(__file__), backup / "train_globe_xtts_pilot.py")
    prepared = run_output / "prepared"
    prepare_training_dataset(splits, prepared)
    packages = {}
    for package in ("coqui-tts", "coqui-tts-trainer", "torch", "torchaudio", "transformers", "soundfile"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    metadata = {
        "status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset": dataset_report, "python": sys.version, "platform": platform.platform(),
        "packages": packages,
        "all_installed_packages": {dist.metadata["Name"]: dist.version for dist in distributions()},
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu (diagnostic override)",
        "output": str(run_output), "launcher_sha256": sha256_file(Path(__file__)),
        "upstream_sha256": sha256_file(Path(gpt_train.__file__)),
        "model_license": "Coqui Public Model License; non-commercial",
    }
    if args.initial_checkpoint:
        initial = args.initial_checkpoint.expanduser().resolve()
        if not initial.is_file():
            parser.error(f"Initial checkpoint missing: {initial}")
        metadata["initial_checkpoint"] = {"path": str(initial), "sha256": sha256_file(initial),
                                          "optimizer_restored": False,
                                          "note": "New full epoch after partial-run model warm-start; not an exact optimizer resume"}
    write_json(metadata_path, metadata)
    original_config_class, original_trainer = gpt_train.GPTTrainerConfig, gpt_train.Trainer

    def pilot_config(**kwargs):
        if not torch.cuda.is_available():
            kwargs["optimizer_params"] = {**kwargs["optimizer_params"], "foreach": False}
        kwargs.update(num_loader_workers=args.workers, print_step=args.print_step,
                      save_step=args.save_step, save_n_checkpoints=2, save_checkpoints=True,
                      num_eval_loader_workers=args.workers, print_eval=True,
                      save_on_interrupt=True, training_seed=args.seed)
        config = original_config_class(**kwargs)
        config.save_json(str(run_output / "training_config.json"))
        return config

    class PilotTrainer(original_trainer):
        def save_checkpoint(self):
            if args.download_snapshots:
                snapshot = run_output / f"weights-step-{self.total_steps_done}.pth"
                report = export_model_snapshot(self._get_model().state_dict(), snapshot,
                                               self.total_steps_done, self.epochs_done)
                report.update(base_assets_sha256=metadata.get("base_assets_sha256"),
                              launcher_sha256=metadata["launcher_sha256"], dataset=metadata["dataset"])
                write_json(snapshot.with_suffix(".json"), report)
                if backup:
                    # Stream the already-created compact file: do not reload
                    # weights or serialize optimizer state on this CPU path.
                    temporary = backup / "latest_model_fp16.pth.partial"
                    shutil.copyfile(snapshot, temporary)
                    temporary.replace(backup / "latest_model_fp16.pth")
                    shutil.copyfile(run_output / "training_config.json", backup / "training_config.json")
                    write_json(backup / "run_metadata.json", metadata)
                    write_json(backup / "snapshot_metadata.json", report)
                    print(f"Compact snapshot staged on mounted storage: {backup}; verify remote sync.", flush=True)
                print(f"WEIGHT_SNAPSHOT_READY {snapshot}", flush=True)
                return
            super().save_checkpoint()
            checkpoint = Path(self.output_path) / f"checkpoint_{self.total_steps_done}.pth"
            if args.download_snapshots:
                snapshot = run_output / f"weights-step-{self.total_steps_done}.pth"
                snapshot_report = export_weight_snapshot(checkpoint, snapshot)
                snapshot_report.update(base_assets_sha256=metadata.get("base_assets_sha256"),
                                       launcher_sha256=metadata["launcher_sha256"], dataset=metadata["dataset"])
                write_json(snapshot.with_suffix(".json"), snapshot_report)
                print(f"WEIGHT_SNAPSHOT_READY {snapshot}", flush=True)
            if backup:
                if not checkpoint.is_file():
                    raise RuntimeError(f"Checkpoint missing before backup: {checkpoint}")
                temporary = backup / "latest_model_fp16.pth.partial"
                snapshot_report = export_weight_snapshot(checkpoint, temporary)
                temporary.replace(backup / "latest_model_fp16.pth")
                shutil.copyfile(run_output / "training_config.json", backup / "training_config.json")
                write_json(backup / "run_metadata.json", metadata)
                write_json(backup / "snapshot_metadata.json", snapshot_report)
                print(f"Persistent checkpoint saved at step {self.total_steps_done}: {backup}", flush=True)

        def __init__(self, *positional, **kwargs):
            if args.initial_checkpoint:
                positional[0].restore_path = str(args.initial_checkpoint.expanduser().resolve())
            super().__init__(*positional, **kwargs)
            if not torch.cuda.is_available():
                enable_memory_saving(self._get_model().xtts.gpt.gpt)
                metadata["memory_optimizations"] = {"gradient_checkpointing": "non-reentrant",
                                                     "adamw_foreach": False,
                                                     "periodic_checkpoints": "live FP16 model weights; no optimizer"}
            metadata["trainer_output"] = str(self.output_path)
            assets = Path(self.config.model_args.xtts_checkpoint).parent
            metadata["base_assets_sha256"] = {
                path.name: sha256_file(path) for path in sorted(assets.iterdir()) if path.is_file()
            }
            metadata["status"] = "training"
            write_json(metadata_path, metadata)

        def save_best_model(self):
            if torch.cuda.is_available() and not args.download_snapshots:
                return super().save_best_model()
            # Keep full-precision best weights for the validated inference
            # export, while avoiding unnecessary optimizer serialization.
            optimizer = self.optimizer
            try:
                self.optimizer = None
                return super().save_best_model()
            finally:
                self.optimizer = optimizer

        def _correct_prompt_mask(self, batch):
            if batch.get("cond_idxs") is not None:
                gpt = self._get_model().xtts.gpt
                if gpt.use_perceiver_resampler:
                    batch["cond_idxs"] = correct_prompt_indices(
                        batch["cond_idxs"], gpt.code_stride_len, gpt.perceiver_cond_length_compression
                    )

        def eval_step(self, batch, *positional, **kwargs):
            self._correct_prompt_mask(batch)
            outputs, losses = super().eval_step(batch, *positional, **kwargs)
            require_finite(losses, "validation loss")
            return outputs, losses

        def eval_epoch(self):
            # Preserve trained weights before a potentially slow validation
            # pass, so a runtime expiry cannot erase the whole epoch.
            if args.download_snapshots:
                self.save_checkpoint()
            print("Starting validation with visible batch logs.", flush=True)
            return super().eval_epoch()

        def _model_train_step(self, batch, criterion, optimizer_idx=None):
            self._correct_prompt_mask(batch)
            outputs, losses = super()._model_train_step(batch, criterion, optimizer_idx)
            try:
                require_finite(losses, f"training loss at step {self.total_steps_done}")
            except FloatingPointError:
                # Preserve the failing input identity for diagnosis, not a
                # checkpoint silently contaminated by an invalid gradient.
                metadata["non_finite_batch"] = {
                    key: str(value) for key, value in batch.items()
                    if key in {"audio_file", "audio_unique_name", "wav_lengths", "text_lengths", "cond_idxs"}
                }
                write_json(metadata_path, metadata)
                raise
            return outputs, losses

        def fit(self):
            # The upstream public fit() calls os._exit on interrupt. Use its
            # version-pinned epoch loop so failures reach our metadata handler.
            try:
                self._fit()
                # _fit saves best_model after validation each epoch. Also keep
                # final optimizer state for runs shorter than save_step.
                self.save_checkpoint()
            except KeyboardInterrupt:
                self.save_checkpoint()
                raise
            finally:
                self.dashboard_logger.finish()

    gpt_train.GPTTrainerConfig, gpt_train.Trainer = pilot_config, PilotTrainer
    print(f"Training {len(splits['train'])} clips; run metadata: {metadata_path}", flush=True)
    try:
        config_path, _, vocab_path, trainer_path, _ = gpt_train.train_gpt(
            "en", args.epochs, args.batch_size, args.grad_accum,
            str(prepared / "train.csv"), str(prepared / "eval.csv"), str(run_output),
        )
        metadata["inference_export"] = export_inference(
            Path(trainer_path), Path(config_path), Path(vocab_path), run_output / "inference"
        )
        metadata["status"] = "completed"
        if backup:
            shutil.copytree(run_output / "inference", backup / "inference")
        print(f"Training complete. Inference files: {run_output / 'inference'}", flush=True)
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        gpt_train.GPTTrainerConfig, gpt_train.Trainer = original_config_class, original_trainer
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(metadata_path, metadata)
        if backup:
            write_json(backup / "run_metadata.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
