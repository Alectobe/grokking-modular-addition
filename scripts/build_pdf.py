"""
Сборка PDF-отчёта: report/REPORT.md -> HTML (pandoc, формулы в MathML) -> PDF (Chromium, A4).

Оформление: титульный лист, содержание с номерами страниц, шрифт Liberation Serif
(метрический аналог Times New Roman) 14 pt, полуторный интервал, поля 30/15/20/20 мм,
подписи «Рисунок N — …» и «Таблица N — …», номер страницы внизу по центру.
Данные титульного листа и подписи — в report/pdf/meta.yaml, вёрстка — в report/pdf/style.css.

Содержание собирается в несколько проходов: после печати номера страниц заголовков
берутся из закладок PDF и подставляются в содержание, пока номера не перестанут меняться.

Нужны pandoc и Chromium (путь — в аргументе --chromium или переменной окружения CHROMIUM).

Пример:
    python scripts/build_pdf.py --out report/report.pdf
"""
import argparse
import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>figures/[^)]+)\)\s*$")
TABLE_START = re.compile(r"^<!-- TABLE:(?P<name>[\w-]+) -->$")
TABLE_END = "<!-- /TABLE -->"
TABLE_RULE = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")
HEADING = re.compile(r'<h(?P<level>[23]) id="(?P<id>[^"]+)"[^>]*>(?P<inner>.*?)</h(?P=level)>', re.DOTALL)
MAX_PASSES = 4


def find_chromium(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("CHROMIUM"), DEFAULT_CHROMIUM,
                  shutil.which("chromium"), shutil.which("chromium-browser"), shutil.which("google-chrome")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Chromium не найден: укажите путь в --chromium или в переменной окружения CHROMIUM")


def table_caption(header_line: str, table_name: str | None, meta: dict) -> str:
    """Подпись таблицы: по имени метки TABLE или по первой ячейке заголовка ручной таблицы."""
    if table_name is not None:
        if table_name not in meta["tables"]:
            raise SystemExit(f"нет подписи для таблицы {table_name} в report/pdf/meta.yaml")
        return meta["tables"][table_name]
    first_cell = header_line.strip().strip("|").split("|")[0].strip()
    if first_cell not in meta["manual_tables"]:
        raise SystemExit(f"нет подписи для таблицы с заголовком «{first_cell}» в report/pdf/meta.yaml")
    return meta["manual_tables"][first_cell]


def preprocess(markdown: str, meta: dict, figures_dir: Path) -> str:
    """Готовит Markdown к вёрстке.

    Шапка отчёта уходит на титульный лист; к рисункам и таблицам добавляются подписи;
    широкие таблицы оборачиваются в блоки с особой вёрсткой; после аннотации добавляется
    ссылка на репозиторий.
    """
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## "))
    lines = lines[start:]

    out: list[str] = []
    table_name = None
    i = 0
    while i < len(lines):
        line = lines[i]
        marker = TABLE_START.match(line.strip())
        if marker:
            table_name = marker.group("name")
            out.append(line)
            i += 1
            continue
        if line.strip() == TABLE_END:
            table_name = None
            out.append(line)
            i += 1
            continue
        image = IMAGE.match(line)
        if image:
            filename = Path(image.group("path")).name
            if filename not in meta["figures"]:
                raise SystemExit(f"нет подписи для рисунка {filename} в report/pdf/meta.yaml")
            out.append(f"![{meta['figures'][filename]}]({(figures_dir / filename).as_posix()})")
            i += 1
            continue
        # Таблица — строка с «|» в начале, за которой идёт строка-разделитель «|---|…»;
        # строка абзаца вида «|log₁₀(…)|: …» таблицей не считается
        if line.startswith("|") and i + 1 < len(lines) and TABLE_RULE.match(lines[i + 1]):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            caption = table_caption(block[0], table_name, meta)
            css_class = None
            if table_name in meta.get("landscape_tables", []):
                css_class = "landscape"
            elif table_name in meta.get("compact_tables", []):
                css_class = "compact"
            if css_class:
                out += ["", f"::: {{.{css_class}}}"]
            # Подпись — отдельный абзац над таблицей: так она не отрывается от таблицы при переносе
            out += raw_html(f'<p class="table-caption">{html.escape(caption)}</p>') + block + [""]
            if css_class:
                out += [":::", ""]
            continue
        out.append(line)
        i += 1

    # Ссылка на репозиторий — в конце аннотации (перед первым разделителем после неё)
    text = "\n".join(out)
    url = html.escape(meta["repository"])
    repo = "\n".join(raw_html(f'<p class="repo">Код, конфигурации и метрики всех запусков: <a href="{url}">{url}</a></p>'))
    first_rule = text.index("\n---\n")
    return text[:first_rule] + f"\n{repo}\n" + text[first_rule:]


def raw_html(fragment: str) -> list[str]:
    """Блок HTML, который pandoc передаёт в результат без разбора Markdown внутри."""
    return ["", "```{=html}", fragment, "```", ""]


def short_ids(body: str) -> str:
    """Короткие латинские id заголовков.

    pandoc строит id из текста заголовка; длинные кириллические id превращаются в имена
    закладок PDF длиннее допустимых 127 байт, и некоторые программы просмотра на них ругаются.
    """
    mapping: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        new_id = f"sec-{len(mapping) + 1}"
        mapping[match.group("id")] = new_id
        return match.group(0).replace(f'id="{match.group("id")}"', f'id="{new_id}"', 1)

    body = HEADING.sub(rename, body)
    for old_id, new_id in mapping.items():
        body = body.replace(f'href="#{old_id}"', f'href="#{new_id}"')
    return body


def pandoc_html(markdown: str) -> str:
    result = subprocess.run(
        ["pandoc", "-f", "markdown-superscript-subscript+autolink_bare_uris", "-t", "html5", "--mathml", "--wrap=none",
         # иначе pandoc задаёт таблицам с длинными строками ширины столбцов по числу дефисов в разделителе
         "--columns=100000"],
        input=markdown, capture_output=True, text=True, check=True,
    )
    return result.stdout


def headings(body: str) -> list[dict]:
    """Заголовки второго и третьего уровня в порядке появления."""
    found = []
    for match in HEADING.finditer(body):
        title = html.unescape(re.sub(r"<[^>]+>", "", match.group("inner"))).strip()
        found.append({"level": int(match.group("level")), "id": match.group("id"), "title": title})
    return found


def title_page(meta: dict) -> str:
    esc = html.escape
    students = "\n".join(f"<p>{esc(name)}</p>" for name in meta["students"])
    return f"""<div class="title-page">
  <p class="university">{esc(meta['university'])}</p>
  <p class="discipline">Дисциплина «{esc(meta['discipline'])}»</p>
  <div class="middle">
    <p class="work-type">{esc(meta['work_type'])}</p>
    <p class="theme-label">на тему</p>
    <p class="theme">«{esc(meta['title'])}»</p>
  </div>
  <div class="people">
    <p>{esc(meta['students_label'])}</p>
    {students}
    <p class="label">{esc(meta['teacher_label'])}</p>
    <p>{esc(meta['teacher'])}</p>
  </div>
  <div class="bottom"><p>{esc(meta['city_year'])}</p></div>
</div>"""


def toc(entries: list[dict], pages: dict[str, int] | None) -> str:
    items = []
    for entry in entries:
        page = pages.get(entry["id"], "00") if pages else "00"
        items.append(
            f'<li class="level-{entry["level"]}"><a class="entry" href="#{entry["id"]}">{html.escape(entry["title"])}</a>'
            f'<span class="dots"></span><span class="page">{page}</span></li>'
        )
    return '<section class="toc"><h2 class="toc-title">Содержание</h2><ol>\n' + "\n".join(items) + "\n</ol></section>"


def document(meta: dict, css: str, body: str, entries: list[dict], pages: dict[str, int] | None) -> str:
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{html.escape(meta['title'])}</title>
<style>
{css}
</style>
</head>
<body>
{title_page(meta)}
{toc(entries, pages)}
{body}
</body>
</html>
"""


def render(chromium: str, html_path: Path, pdf_path: Path) -> None:
    command = [
        chromium, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
        "--generate-pdf-document-outline", "--virtual-time-budget=20000",
        f"--print-to-pdf={pdf_path}", html_path.as_uri(),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=300)


def normalize(text: str) -> str:
    return " ".join(text.replace(" ", " ").split())


def heading_pages(pdf_path: Path, entries: list[dict]) -> dict[str, int]:
    """Номера страниц заголовков по закладкам PDF, которые Chromium строит по заголовкам."""
    reader = PdfReader(pdf_path)
    outline: list[tuple[str, int]] = []

    def walk(items) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
            else:
                outline.append((normalize(item.title), reader.get_destination_page_number(item) + 1))

    walk(reader.outline)
    pages, position = {}, 0
    for entry in entries:
        target = normalize(entry["title"])
        while position < len(outline) and outline[position][0] != target:
            position += 1
        if position == len(outline):
            raise SystemExit(f"заголовок «{entry['title']}» не найден в закладках PDF")
        pages[entry["id"]] = outline[position][1]
        position += 1
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка PDF-отчёта")
    parser.add_argument("--report", type=Path, default=ROOT / "report" / "REPORT.md")
    parser.add_argument("--meta", type=Path, default=ROOT / "report" / "pdf" / "meta.yaml")
    parser.add_argument("--css", type=Path, default=ROOT / "report" / "pdf" / "style.css")
    parser.add_argument("--out", type=Path, default=ROOT / "report" / "report.pdf")
    parser.add_argument("--chromium", type=str, default=None)
    parser.add_argument("--keep-html", type=Path, default=None, help="куда сохранить промежуточный HTML")
    args = parser.parse_args()

    # pypdf предупреждает о мелких несоответствиях в PDF от Chromium (размер trailer) — на результат не влияет
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    chromium = find_chromium(args.chromium)
    meta = yaml.safe_load(args.meta.read_text(encoding="utf-8"))
    css = args.css.read_text(encoding="utf-8")
    figures_dir = args.report.resolve().parent / "figures"
    body = short_ids(pandoc_html(preprocess(args.report.read_text(encoding="utf-8"), meta, figures_dir)))
    entries = headings(body)

    with tempfile.TemporaryDirectory() as tmp:
        html_path, pdf_path = Path(tmp) / "report.html", Path(tmp) / "report.pdf"
        pages = None
        for attempt in range(1, MAX_PASSES + 1):
            html_path.write_text(document(meta, css, body, entries, pages), encoding="utf-8")
            render(chromium, html_path, pdf_path)
            new_pages = heading_pages(pdf_path, entries)
            if new_pages == pages:
                break
            pages = new_pages
        else:
            raise SystemExit("номера страниц в содержании не стабилизировались")
        if args.keep_html:
            shutil.copyfile(html_path, args.keep_html)

        writer = PdfWriter(clone_from=str(pdf_path))
        writer.add_metadata({
            "/Title": meta["title"],
            "/Author": ", ".join(meta["students"]),
            "/Subject": f"{meta['work_type']} по дисциплине «{meta['discipline']}», {meta['university']}",
        })
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as f:
            writer.write(f)

    n_pages = len(PdfReader(args.out).pages)
    print(f"готово: {args.out} — {n_pages} стр., проходов вёрстки: {attempt}")


if __name__ == "__main__":
    main()
