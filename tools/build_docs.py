"""Convert the markdown appendices into house-style HTML pages.

Run once to migrate, and again whenever a page needs regenerating from a markdown draft. After the
migration the HTML files ARE the source -- there is no markdown left to regenerate them from, which
is deliberate: two copies of a document drift, and the one people read is the one that should be
edited.

    python tools/build_docs.py <file.md> [...]        # writes <file>.html beside each

The style is lifted verbatim from blueprint_scopex.html so the whole doc set reads as one thing,
and `.md` links are rewritten to `.html` so cross-references keep working.
"""

from __future__ import annotations

import pathlib
import re
import sys

import markdown

HERE = pathlib.Path(__file__).resolve().parent.parent
BLUEPRINT = HERE / "docs" / "blueprint_scopex.html"


def house_style() -> str:
    """The <style> block from the blueprint, so there is exactly one stylesheet in the project."""
    s = BLUEPRINT.read_text()
    return s[s.index("<style>"):s.index("</style>") + len("</style>")]


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — scopex</title>
{style}
<style>
  main{{padding:46px 52px 140px;max-width:1000px;margin:0 auto}}
  .backlink{{font:600 12px/1.4 var(--mono);letter-spacing:.12em;text-transform:uppercase;
             color:var(--accent);text-decoration:none;display:inline-block;margin-bottom:18px}}
  .aud{{font-size:15px;color:var(--muted);border-left:3px solid var(--accent);padding:8px 0 8px 16px;
        margin:14px 0 28px;background:var(--accent-soft);border-radius:0 8px 8px 0}}
  blockquote{{border-left:3px solid var(--rule);margin:16px 0;padding:2px 0 2px 18px;color:var(--muted)}}
  h1{{font-size:32px;line-height:1.15;margin:6px 0 10px;letter-spacing:-.01em}}
</style>
<body>
<main>
<a class="backlink" href="index.html">&larr; scopex docs</a>
{body}
<div class="footer">Part of the <a href="blueprint_scopex.html">scopex blueprint</a>.
Every number measured on jax 0.10.2.</div>
</main>
</body></html>
"""


def convert(src: pathlib.Path) -> pathlib.Path:
    text = src.read_text()
    html = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])

    # cross-references between appendices must follow them to .html
    html = re.sub(r'href="([^"]+)\.md"', r'href="\1.html"', html)
    html = html.replace('href="README.html"', 'href="index.html"')

    # the audience line each file opens with becomes a styled banner
    html = re.sub(r"<p><strong>(Audience:[^<]*)</strong></p>",
                  r'<div class="aud"><strong>\1</strong></div>', html, count=1)
    html = re.sub(r"<p><strong>(For someone[^<]*)</strong></p>",
                  r'<div class="aud"><strong>Audience: \1</strong></div>', html, count=1)

    title = re.search(r"<h1[^>]*>(.*?)</h1>", html)
    out = src.with_suffix(".html")
    out.write_text(PAGE.format(
        title=re.sub(r"<[^>]+>", "", title.group(1)) if title else src.stem,
        style=house_style(), body=html))
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        p = convert(pathlib.Path(a))
        print(f"  {a} -> {p}  ({p.stat().st_size // 1024} KB)")
