"""Вывод таблиц для отчёта: CSV для машин и Markdown для чтения."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


def format_value(value, digits: int = 3) -> str:
    """Число для таблицы: целые без дробной части, малые — в научной записи, inf — '> T'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, float) and math.isinf(value):
        return "> T"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.2e}"
        if abs(value) >= 100 or (float(value).is_integer() and abs(value) >= 10):
            return str(round(value))  # эпохи и нормы: дробная часть не нужна
        return f"{value:.{digits}g}"
    return str(value)


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(format_value(v, digits) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def save_table(df: pd.DataFrame, stem: Path, digits: int = 3) -> None:
    """Сохранить таблицу как <stem>.csv и <stem>.md."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(markdown_table(df, digits) + "\n", encoding="utf-8")


def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
