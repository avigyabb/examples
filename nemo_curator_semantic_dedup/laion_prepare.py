# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import os
from typing import Sequence

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def download_shards(repo_id: str, shards: Sequence[str], out_dir: str, token: str | None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    local_paths: list[str] = []
    for shard in shards:
        src = hf_hub_download(repo_id=repo_id, filename=shard, repo_type="dataset", token=token)
        dst = os.path.join(out_dir, shard)
        if src != dst and not os.path.exists(dst):
            import shutil

            shutil.copy(src, dst)
        elif src == dst:
            # Already in place (e.g., cache resolved inside out_dir)
            pass
        local_paths.append(dst if os.path.exists(dst) else src)
        print(f"Downloaded {shard} -> {local_paths[-1]}")
    return local_paths


def build_subset(
    shard_paths: Sequence[str],
    out_parquet: str,
    max_rows: int,
    columns: Sequence[str] = ("URL", "TEXT"),
) -> None:
    writer = None
    rows_written = 0

    for path in sorted(shard_paths):
        schema = pq.read_schema(path)
        available = set(schema.names)

        # Resolve columns against available names (handle lower/upper and LAION's caption)
        resolved_cols: list[str] = []
        for col in columns:
            if col in available:
                resolved_cols.append(col)
                continue
            lower = col.lower()
            if lower in available:
                resolved_cols.append(lower)
                continue
            # Special-case mapping TEXT -> caption
            if col.upper() == "TEXT" and "caption" in available:
                resolved_cols.append("caption")

        if not resolved_cols:
            raise ValueError(f"None of requested columns {columns} found in {path}; available: {sorted(available)}")

        table = pq.read_table(path, columns=resolved_cols)
        if max_rows and rows_written + table.num_rows > max_rows:
            table = table.slice(0, max_rows - rows_written)

        if writer is None:
            os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
            writer = pq.ParquetWriter(out_parquet, table.schema)

        writer.write_table(table)
        rows_written += table.num_rows
        print(f"Wrote {rows_written} rows so far")
        if max_rows and rows_written >= max_rows:
            break

    if writer:
        writer.close()
        print(f"Final output: {out_parquet}, rows: {rows_written}")
    else:
        print("No data written (no shards?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected LAION-400M metadata shards and build a combined Parquet "
            "with URL/TEXT columns, optionally capped at max_rows."
        )
    )
    parser.add_argument(
        "--repo-id",
        default="laion/laion400m-met-release",
        help="HF dataset repo id for LAION metadata.",
    )
    parser.add_argument(
        "--shards",
        default="part-00000-4227e361-38e7-40d5-8822-c6db46ea077c-c000.snappy.parquet",
        help=(
            "Comma-separated shard filenames to download. Use exact names from the repo, e.g., "
            "part-00000-4227e361-38e7-40d5-8822-c6db46ea077c-c000.snappy.parquet"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="./laion_meta",
        help="Directory to store downloaded shards.",
    )
    parser.add_argument(
        "--out-parquet",
        default="./laion_meta/laion_subset.parquet",
        help="Output Parquet path for concatenated subset.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on total rows (0 means keep all rows from provided shards).",
    )
    parser.add_argument(
        "--columns",
        default="URL,TEXT",
        help="Comma-separated columns to keep (default: URL,TEXT).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated/private datasets (or set HF_TOKEN env var).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = [s.strip() for s in args.shards.split(",") if s.strip()]
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    token = args.hf_token or os.environ.get("HF_TOKEN")

    shard_paths = download_shards(args.repo_id, shards, args.out_dir, token=token)
    build_subset(shard_paths, args.out_parquet, args.max_rows, columns=columns)


if __name__ == "__main__":
    main()

