import math
import sys
from pathlib import Path

import pandas as pd

from grokking.report import format_value, markdown_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fill_report import fill


def test_format_value():
    assert format_value(None) == "—"
    assert format_value(math.nan) == "—"
    assert format_value(math.inf) == "> T"
    assert format_value(1815.0) == "1815"
    assert format_value(0.123456) == "0.123"
    assert format_value(3e-21) == "3.00e-21"


def test_markdown_table():
    table = markdown_table(pd.DataFrame({"a": [1, 2], "b": [0.5, math.inf]}))
    assert table.splitlines() == ["| a | b |", "|---|---|", "| 1 | 0.5 |", "| 2 | > T |"]


def test_fill_replaces_block_and_is_idempotent(tmp_path):
    (tmp_path / "demo.md").write_text("| x |\n|---|\n| 1 |\n", encoding="utf-8")
    text = "до\n<!-- TABLE:demo -->\nстарое\n<!-- /TABLE -->\nпосле"
    once, used = fill(text, tmp_path)
    twice, _ = fill(once, tmp_path)
    assert used == ["demo"]
    assert once == twice == "до\n<!-- TABLE:demo -->\n| x |\n|---|\n| 1 |\n<!-- /TABLE -->\nпосле"
