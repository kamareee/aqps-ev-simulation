"""
Assemble the EPSR LaTeX submission from the manuscript sources and the generated paper outputs.

Sources : manuscript/*.tex, refs.bib, elsarticle.cls, elsarticle-num-names.bst, epsr.sty (this folder)
Outputs : paper/numbers.tex, paper/tables/tab_*.tex, paper/figures/*.pdf (postprocess/build.py)
Result  : manuscript/build/AQPS_EPSR/            flat folder (Editorial Manager accepts no subfolders)
          manuscript/build/AQPS_EPSR_EM.zip       files needed to compile main.tex (upload to Editorial Manager)
          manuscript/build/AQPS_EPSR_Overleaf.zip same plus highlights, cover letter, number map
Only the tables and figures that main.tex uses are copied. Compiles with latexmk if it is installed.

Run from the repo root:  uv run --frozen python manuscript/assemble.py
"""

import csv
import os
import re
import shutil
import subprocess
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "manuscript")
PAPER = os.path.join(ROOT, "paper")
BUILD = os.path.join(SRC, "build")
OUT = os.path.join(BUILD, "AQPS_EPSR")
CORE = ["main.tex", "refs.bib", "elsarticle.cls", "elsarticle-num-names.bst", "epsr.sty"]
EXTRA = ["highlights.tex", "cover_letter.tex", "change_list.md"]


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    for f in CORE + EXTRA:
        shutil.copy(os.path.join(SRC, f), OUT)
    shutil.copy(os.path.join(PAPER, "numbers.tex"), OUT)
    tex = open(os.path.join(SRC, "main.tex"), encoding="utf-8").read()
    tables = [t + ".tex" for t in re.findall(r"\\input\{(tab_[^}]*)\}", tex)]
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    for t in tables:
        shutil.copy(os.path.join(PAPER, "tables", t), OUT)
    for f in figures:
        shutil.copy(os.path.join(PAPER, "figures", f), OUT)
    # number -> source map restricted to the macros used in the submission files
    used = set()
    for f in ["main.tex", "highlights.tex", "cover_letter.tex"]:
        used |= set(re.findall(r"\\R[A-Za-z]+", open(os.path.join(SRC, f), encoding="utf-8").read()))
    with open(os.path.join(PAPER, "numbers_map.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = sorted(used - {r["macro"] for r in rows} - {"\\Return"})
    if missing:
        raise SystemExit(f"macros used but not defined in paper/numbers.tex: {missing}")
    with open(os.path.join(OUT, "numbers_used.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(r for r in rows if r["macro"] in used)
    em = CORE + ["numbers.tex"] + tables + figures
    with zipfile.ZipFile(os.path.join(BUILD, "AQPS_EPSR_EM.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in em:
            z.write(os.path.join(OUT, f), f)
    with zipfile.ZipFile(os.path.join(BUILD, "AQPS_EPSR_Overleaf.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(OUT)):
            z.write(os.path.join(OUT, f), f)
    print(f"  {len(tables)} tables, {len(figures)} figures, {len(used)} number macros -> {OUT}")
    if shutil.which("latexmk"):
        for f in ("main.tex", "highlights.tex", "cover_letter.tex"):
            r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", f], cwd=OUT,
                               capture_output=True, text=True)
            print(f"  latexmk {f}: {'ok' if r.returncode == 0 else 'FAILED (see log)'}")


if __name__ == "__main__":
    main()
