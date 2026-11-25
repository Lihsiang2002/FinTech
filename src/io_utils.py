"""共用檔案 I/O 工具模組（原子寫入與壓縮一致性）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import MARKET_ETL_CFG


def atomic_write_parquet(
    data: Union[pa.Table, pd.DataFrame],
    target_path: Path,
    compression: str = MARKET_ETL_CFG.parquet_compression,
) -> Path:
    """以「暫存寫入 + 原子替換」方式寫入 Parquet，防範寫入中斷導致檔案毀損。

    Args:
        data: PyArrow Table 或 Pandas DataFrame。
        target_path: 最終目標 Parquet 檔案路徑。
        compression: 壓縮演算法（預設 zstd）。

    Returns:
        最終寫入完成之目標檔案路徑。
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")

    table = (
        data
        if isinstance(data, pa.Table)
        else pa.Table.from_pandas(data, preserve_index=False)
    )
    try:
        pq.write_table(table, tmp_path, compression=compression)
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise IOError(f"暫存 Parquet 檔寫入異常（檔案不存在或大小為 0）：{tmp_path}")
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return target_path
