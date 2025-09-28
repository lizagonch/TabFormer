#!/usr/bin/env python3
"""Validate that TabFormer datasets were created successfully."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dataset.card import TransactionDataset
from dataset.prsa import PRSADataset


LOGGER = logging.getLogger("check_dataset")


class DatasetCheckError(RuntimeError):
    """Raised when a dataset fails validation."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-type",
        choices=["card", "prsa"],
        default="card",
        help="Dataset to validate.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data/credit_card/"),
        help="Location of the dataset artefacts.",
    )
    parser.add_argument(
        "--data-fname",
        default="card_transaction.v1",
        help="Base file name of the credit card dataset (ignored for PRSA).",
    )
    parser.add_argument(
        "--data-extension",
        default="",
        help="Optional suffix used during preprocessing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing the generated vocabulary and summary file.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Window length to use when instantiating the dataset for checks.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Stride to use when instantiating the dataset for checks.",
    )
    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Construct the dataset in masked-language-modelling mode.",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Load the flattened representation of the dataset.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Explicit path to the summary JSON file. Defaults to <output-dir>/<dataset>_dataset_summary.json",
    )
    return parser


def _require_files_exist(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise DatasetCheckError(f"Missing expected files: {', '.join(str(path) for path in missing)}")


def _load_summary(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    summary_path = args.summary_file
    if summary_path is None:
        summary_path = args.output_dir / f"{args.data_type}_dataset_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    LOGGER.warning("Summary file not found at %s", summary_path)
    return None


def _check_card_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    preprocessed_dir = args.data_root / "preprocessed"
    suffix = f"_{args.data_extension}" if args.data_extension else ""
    expected_files = [
        preprocessed_dir / f"{args.data_fname}{suffix}.encoded.csv",
        preprocessed_dir / f"{args.data_fname}{suffix}.encoder_fit.pkl",
        args.output_dir / f"vocab{suffix}.nb",
    ]
    _require_files_exist(expected_files)

    dataset = TransactionDataset(
        mlm=args.mlm,
        seq_len=args.seq_len,
        stride=args.stride,
        cached=True,
        root=str(args.data_root),
        fname=args.data_fname,
        vocab_dir=str(args.output_dir),
        fextension=args.data_extension,
        flatten=args.flatten,
        return_labels=True,
    )

    summary = _load_summary(args)
    if summary is not None and summary.get("num_samples") != len(dataset):
        raise DatasetCheckError(
            f"Sample count mismatch: summary has {summary.get('num_samples')} but dataset returned {len(dataset)}"
        )

    return {
        "num_samples": len(dataset),
        "num_columns": dataset.ncols,
        "vocab_size": len(dataset.vocab.id2token),
        "fraudulent_windows": int(sum(dataset.window_label)),
        "summary": summary,
    }


def _check_prsa_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    expected_files = [args.output_dir / "vocab.nb"]
    _require_files_exist(expected_files)

    dataset = PRSADataset(
        data_root=str(args.data_root),
        seq_len=args.seq_len,
        stride=args.stride,
        vocab_dir=str(args.output_dir),
        mlm=args.mlm,
        flatten=args.flatten,
    )

    summary = _load_summary(args)
    if summary is not None and summary.get("num_samples") != len(dataset):
        raise DatasetCheckError(
            f"Sample count mismatch: summary has {summary.get('num_samples')} but dataset returned {len(dataset)}"
        )

    return {
        "num_samples": len(dataset),
        "num_columns": dataset.ncols,
        "vocab_size": len(dataset.vocab.id2token),
        "summary": summary,
    }


def main(raw_args: Optional[Any] = None) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _build_parser().parse_args(raw_args)
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.data_type == "card":
        report = _check_card_dataset(args)
    else:
        report = _check_prsa_dataset(args)

    LOGGER.info("Dataset validation report:\n%s", json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except DatasetCheckError as exc:
        LOGGER.error("Dataset validation failed: %s", exc)
        raise
