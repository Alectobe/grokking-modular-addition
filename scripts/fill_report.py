"""
Вставить таблицы из report/tables/*.md в отчёт между метками
    <!-- TABLE:имя -->
    ...
    <!-- /TABLE -->
Повторный запуск заменяет содержимое между метками, поэтому после пересчёта анализа
таблицы в отчёте обновляются одной командой.

Пример:
    python scripts/fill_report.py --report report/REPORT.md --tables report/tables
"""
import argparse
import re
from pathlib import Path

BLOCK = re.compile(r"(<!-- TABLE:(?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- /TABLE -->)", re.DOTALL)


def fill(text: str, tables: Path) -> tuple[str, list[str]]:
    used = []

    def replace(match: re.Match) -> str:
        name = match.group("name")
        table = (tables / f"{name}.md").read_text(encoding="utf-8").strip()
        used.append(name)
        return f"{match.group(1)}{table}\n{match.group(4)}"

    return BLOCK.sub(replace, text), used


def main() -> None:
    parser = argparse.ArgumentParser(description="Вставка таблиц анализа в отчёт")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()
    text, used = fill(args.report.read_text(encoding="utf-8"), args.tables)
    args.report.write_text(text, encoding="utf-8")
    print(f"обновлено таблиц: {len(used)} ({', '.join(used)})")


if __name__ == "__main__":
    main()
