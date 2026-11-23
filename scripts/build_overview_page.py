#!/usr/bin/env python3
"""Render docs/project-overview.md as one self-contained HTML page, figures embedded.

Images in the Markdown (paths relative to the Markdown file) are read and embedded as data URIs, so the page opens on its
own. Headings get anchors and a table of contents in a side column on wide screens.

    python scripts/build_overview_page.py [--source docs/project-overview.md] [--out docs/project-overview.html]
"""

from __future__ import annotations

import argparse
import base64
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import markdown

CSS = """
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6b6b70; --line: #e3e3e6; --accent: #1a56b0; --soft: #f5f6f8; }
@media (prefers-color-scheme: dark) { :root { --bg: #16171a; --fg: #e8e8ea; --muted: #a0a0a8; --line: #33343a; --accent: #7fa7e0; --soft: #202126; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.75 "Pretendard", "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", system-ui, sans-serif; }
.wrap { display: grid; grid-template-columns: 260px minmax(0, 860px); gap: 48px; max-width: 1220px; margin: 0 auto; padding: 40px max(16px, env(safe-area-inset-left, 0px)) 80px; }
nav { position: sticky; top: 24px; align-self: start; max-height: calc(100vh - 48px); overflow: auto; font-size: 13.5px; line-height: 1.5; }
nav a { display: block; color: var(--muted); text-decoration: none; padding: 3px 0; }
nav a.h3 { padding-left: 14px; font-size: 12.5px; }
nav a:hover { color: var(--accent); }
main { min-width: 0; }
h1 { font-size: 30px; line-height: 1.3; margin: 0 0 8px; }
h2 { font-size: 23px; margin: 56px 0 12px; padding-top: 12px; border-top: 1px solid var(--line); }
h3 { font-size: 18px; margin: 36px 0 8px; }
p, li { text-wrap: pretty; }
strong { font-weight: 650; }
code { font: 13.5px/1.5 "JetBrains Mono", ui-monospace, monospace; background: var(--soft); padding: 1px 5px; border-radius: 4px; }
pre { background: var(--soft); padding: 14px 16px; border-radius: 8px; overflow-x: auto; }
pre code { background: none; padding: 0; }
blockquote { margin: 16px 0; padding: 4px 18px; border-left: 3px solid var(--line); color: var(--muted); }
.table { overflow-x: auto; margin: 16px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; line-height: 1.5; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; border-bottom: 1.5px solid var(--fg); }
img { max-width: 100%; height: auto; display: block; margin: 18px auto; border-radius: 6px; background: #fff; }
.meta { color: var(--muted); font-size: 13px; }
@media (max-width: 900px) { .wrap { grid-template-columns: 1fr; gap: 0; } nav { display: none; } }
"""


def embed_images(html: str, base: Path) -> str:
    def replace(match: re.Match) -> str:
        source = match.group(1)
        path = (base / source).resolve()
        if not path.exists() or source.startswith(("http:", "https:", "data:")):
            return match.group(0)
        kind = "png" if path.suffix.lower() == ".png" else path.suffix.lower().lstrip(".")
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:image/{kind};base64,{data}"'

    return re.sub(r'src="([^"]+)"', replace, html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("docs/project-overview.md"))
    parser.add_argument("--out", type=Path, default=Path("docs/project-overview.html"))
    args = parser.parse_args()
    text = args.source.read_text()
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"], extension_configs={"toc": {"permalink": False}})
    body = md.convert(text)
    body = re.sub(r"<table>", '<div class="table"><table>', body).replace("</table>", "</table></div>")
    body = embed_images(body, args.source.parent)
    links = []
    for token in md.toc_tokens:
        for h2 in [token] if token["level"] == 2 else token.get("children", []):
            links.append(f'<a href="#{h2["id"]}">{h2["name"]}</a>')
            for h3 in h2.get("children", []):
                links.append(f'<a class="h3" href="#{h3["id"]}">{h3["name"]}</a>')
    title = re.search(r"^# (.+)$", text, re.M).group(1)
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")
    page = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap">
<style>{CSS}</style></head>
<body><div class="wrap"><nav>{''.join(links)}</nav><main>
<p class="meta">Built from {args.source} at {stamp}</p>
{body}
</main></div></body></html>
"""
    args.out.write_text(page)
    print(f"{args.out}: {len(page) // 1024} KB")


if __name__ == "__main__":
    main()
