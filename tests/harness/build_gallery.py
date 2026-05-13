"""Build a single HTML page for listening to bake-off outputs side-by-side.

Layout: one section per reference clip (with the original ref playable at the
top), and within each section a table where rows = test sentences and columns
= models. Each cell is an <audio controls>.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HARNESS = ROOT / "tests" / "harness"
import sys
if "--bundle" in sys.argv:
    BUNDLE = ROOT / "_listen" / "bakeoff"
    REFS_DIR = BUNDLE / "refs"
    OUT_DIR = BUNDLE
    INDEX_HTML = BUNDLE / "index.html"
else:
    REFS_DIR = HARNESS / "refs"
    OUT_DIR = ROOT / "tests" / "outputs" / "harness"
    INDEX_HTML = OUT_DIR / "_index.html"

# Models in the order we want them displayed
MODELS_TIER1 = ["chatterbox", "f5-tts", "xtts-v2", "index-tts-2", "gpt-sovits"]
MODELS_TIER2 = ["cosyvoice", "voxcpm", "fish-speech", "styletts2", "tortoise-tts"]
MODELS_TIER3 = ["bark", "neutts-air", "omnivoice", "rvc", "seed-vc"]
ALL_MODELS = MODELS_TIER1 + MODELS_TIER2 + MODELS_TIER3


def collect():
    refs = []
    for wav in sorted(REFS_DIR.glob("*.wav")):
        ref_id = wav.stem
        text_path = wav.with_suffix(".txt")
        ref_text = text_path.read_text().strip() if text_path.exists() else ""
        refs.append((ref_id, wav, ref_text))
    sentences = json.loads((HARNESS / "sentences.json").read_text())
    return refs, sentences


def rel(p: Path) -> str:
    """Relative path from INDEX_HTML to p, suitable for HTML src."""
    return os.path.relpath(p.resolve(), INDEX_HTML.parent)


def main():
    refs, sentences = collect()

    parts = []
    parts.append("""<!doctype html>
<html lang=en>
<meta charset=utf-8>
<title>Voice cloning bake-off</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1600px; margin: 1em auto; padding: 0 1em; color: #222; }
h1 { margin-bottom: 0.2em; }
h2 { border-bottom: 2px solid #444; padding-bottom: 0.3em; margin-top: 1.5em; }
h2 .meta { font-size: 0.7em; color: #666; font-weight: normal; }
.ref-block { margin: 1em 0; padding: 1em; background: #f7f7f7; border-radius: 8px; }
.ref-block .transcript { font-style: italic; color: #555; margin: 0.3em 0; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
th, td { padding: 6px 10px; text-align: left; vertical-align: top; }
th { background: #eee; font-weight: 600; position: sticky; top: 0; }
th.model-tier1 { background: #d4f1d4; }
th.model-tier2 { background: #d4e4f1; }
th.model-tier3 { background: #f1e4d4; }
tr:nth-child(even) { background: #fafafa; }
td.sentence-cell { font-weight: 600; min-width: 110px; }
td.sentence-cell .text { font-weight: normal; color: #555; font-size: 0.85em; margin-top: 0.3em; }
audio { width: 240px; height: 32px; }
.legend { display: flex; gap: 1em; font-size: 0.9em; margin: 0.5em 0 1em; }
.legend span { padding: 2px 8px; border-radius: 4px; }
.legend .t1 { background: #d4f1d4; }
.legend .t2 { background: #d4e4f1; }
.legend .t3 { background: #f1e4d4; }
nav.toc { background: #fffbe6; padding: 0.5em 1em; border-radius: 8px; margin-bottom: 1em; }
nav.toc a { margin-right: 1em; }
</style>
<body>
<h1>Voice cloning bake-off</h1>
<p>15 models &times; 5 reference clips &times; 3 test sentences = <b>225 outputs</b>.</p>
<div class=legend>
  <span class=t1>Tier 1 (top contenders)</span>
  <span class=t2>Tier 2 (variety picks)</span>
  <span class=t3>Tier 3 (specialized: incl. VC + Bark non-clone)</span>
</div>
<p><b>Test sentences:</b></p>
<ul>
""")
    for sid, text in sentences.items():
        parts.append(f"<li><code>{sid}</code> &mdash; &ldquo;{text}&rdquo;</li>")
    parts.append("</ul>\n")

    parts.append("<nav class=toc>Jump to reference: ")
    for ref_id, _, _ in refs:
        parts.append(f"<a href=\"#{ref_id}\">{ref_id}</a>")
    parts.append("</nav>\n")

    def model_class(m):
        if m in MODELS_TIER1: return "model-tier1"
        if m in MODELS_TIER2: return "model-tier2"
        return "model-tier3"

    for ref_id, ref_wav, ref_text in refs:
        parts.append(f"<h2 id=\"{ref_id}\">{ref_id} <span class=meta>(reference clip)</span></h2>\n")
        parts.append("<div class=ref-block>")
        parts.append(f"<div class=transcript>&ldquo;{ref_text}&rdquo;</div>")
        parts.append(f"<audio controls preload=metadata src=\"{rel(ref_wav)}\"></audio>")
        parts.append("</div>\n")

        parts.append("<table>\n<thead><tr><th>sentence</th>")
        for m in ALL_MODELS:
            parts.append(f"<th class=\"{model_class(m)}\">{m}</th>")
        parts.append("</tr></thead>\n<tbody>\n")

        for sid, text in sentences.items():
            parts.append(f"<tr><td class=sentence-cell><code>{sid}</code><div class=text>&ldquo;{text}&rdquo;</div></td>")
            for m in ALL_MODELS:
                out_path = OUT_DIR / m / f"{ref_id}__{sid}.wav"
                if out_path.exists():
                    parts.append(f"<td><audio controls preload=metadata src=\"{rel(out_path)}\"></audio></td>")
                else:
                    parts.append("<td><em>missing</em></td>")
            parts.append("</tr>\n")
        parts.append("</tbody></table>\n")

    parts.append("</body></html>\n")
    INDEX_HTML.write_text("".join(parts))
    print(f"wrote {INDEX_HTML}  ({INDEX_HTML.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
