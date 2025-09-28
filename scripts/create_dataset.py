#!/usr/bin/env python3
"""Materialise TabFormer datasets using the existing preprocessing pipelines."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dataset.card import TransactionDataset
from dataset.prsa import PRSADataset


LOGGER = logging.getLogger("create_dataset")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-type",
        choices=["card", "prsa"],
        default="card",
        help="Dataset to materialize.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data/credit_card/"),
        help="Location of the raw dataset files.",
    )
    parser.add_argument(
        "--data-fname",
        default="card_transaction.v1",
        help="Base file name of the credit card dataset (ignored for PRSA).",
    )
    parser.add_argument(
        "--data-extension",
        default="",
        help="Optional suffix for identifying cached files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where vocabularies and summaries will be stored.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Window length passed to the dataset constructors.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Sliding window stride for the dataset constructors.",
    )
    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Whether to prepare the dataset for masked language modelling.",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten records so they can be fed directly into GPT-style models.",
    )
    parser.add_argument(
        "--skip-user",
        action="store_true",
        help="Drop the user field when materialising the credit card dataset.",
    )
    parser.add_argument(
        "--user-ids",
        nargs="+",
        default=None,
        help="Optional list of user identifiers to filter by (card dataset only).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit the number of raw rows ingested from the credit card CSV.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help=(
            "Load already processed artefacts if present. Without this flag the "
            "raw files are reprocessed to refresh the cache."
        ),
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of quantisation bins for numerical features in the card dataset.",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=50,
        help="Number of quantisation bins for numerical features in the PRSA dataset.",
    )
    return parser


def _create_card_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = TransactionDataset(
        mlm=args.mlm,
        user_ids=args.user_ids,
        seq_len=args.seq_len,
        num_bins=args.num_bins,
        cached=args.use_cache,
        root=str(args.data_root),
        fname=args.data_fname,
        vocab_dir=str(args.output_dir),
        fextension=args.data_extension,
        nrows=args.nrows,
        flatten=args.flatten,
        stride=args.stride,
        return_labels=True,
        skip_user=args.skip_user,
    )

    fraud_windows = int(sum(dataset.window_label)) if hasattr(dataset, "window_label") else None
    trans_table = dataset.trans_table if hasattr(dataset, "trans_table") else None
    unique_users = int(trans_table["User"].nunique()) if trans_table is not None and "User" in trans_table else None

    return {
        "dataset_type": "card",
        "num_samples": len(dataset),
        "num_columns": dataset.ncols,
        "flatten": args.flatten,
        "mlm": args.mlm,
        "stride": args.stride,
        "seq_len": args.seq_len,
        "vocab_size": len(dataset.vocab.id2token),
        "num_users": unique_users,
        "fraudulent_windows": fraud_windows,
        "cache_used": args.use_cache,
        "data_root": str(args.data_root),
        "data_file": f"{args.data_fname}.csv",
    }


def _create_prsa_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = PRSADataset(
        data_root=str(args.data_root),
        seq_len=args.seq_len,
        stride=args.stride,
        nbins=args.nbins,
        vocab_dir=str(args.output_dir),
        mlm=args.mlm,
        flatten=args.flatten,
    )

    return {
        "dataset_type": "prsa",
        "num_samples": len(dataset),
        "num_columns": dataset.ncols,
        "flatten": args.flatten,
        "mlm": args.mlm,
        "stride": args.stride,
        "seq_len": args.seq_len,
        "vocab_size": len(dataset.vocab.id2token),
        "cache_used": args.use_cache,
        "data_root": str(args.data_root),
    }


def _store_summary(summary: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{summary['dataset_type']}_dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary_path


def main(raw_args: Optional[Any] = None) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _build_parser().parse_args(raw_args)

    if args.data_type == "card":
        summary = _create_card_dataset(args)
    else:
        summary = _create_prsa_dataset(args)

    summary_path = _store_summary(summary, args.output_dir)
    LOGGER.info("Dataset materialisation finished. Summary saved to %s", summary_path)
    LOGGER.info("Summary: \n%s", json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
