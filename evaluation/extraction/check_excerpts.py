"""Automatically check whether each 'source excerpt' in the extraction output really appears in the paper's PDF text.

Usage:
  python check_excerpts.py extraction_results.xlsx --pdf-dir <folder containing the PDFs, searched recursively> [--excerpt-column COLUMN] [--out excerpt_check.csv]
The PDFs must be findable by the Filename in the extraction table (SLR stores uploads under those hashed names).
The source field accepts "verbatim quote || location" or "NOT_FOUND".
Statuses: pass (all segments found), pass_fragmented (all fragments found in order nearby),
partial, fail (no segment matched), language_mismatch,
no_excerpt, too_short, pdf_missing, not_found, location_only.
"""
import argparse
import pandas as pd
from extraction_utils import (read_table, pick_col, norm_col, load_pdf_text_and_pages, index_pdfs,
                              parse_excerpt_field, check_excerpt_with_pages, wilson)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extraction"); ap.add_argument("--pdf-dir", required=True)
    ap.add_argument(
        "--excerpt-column", "--col-excerpt", metavar="COLUMN",
        default="source_section_table_or_text_excerpt",
        help="Column containing the source quote (default: %(default)s)",
    )
    ap.add_argument("--out", default="excerpt_check.csv")
    a = ap.parse_args()
    df = read_table(a.extraction)
    fcol = pick_col(df, "filename", ["filename"], "filename")
    ecol = norm_col(a.excerpt_column)
    if ecol not in df.columns:
        available_columns = ", ".join(df.columns)
        ap.error(f"excerpt column {a.excerpt_column!r} not found; available columns: {available_columns}")
    idx = index_pdfs(a.pdf_dir); cache = {}; rows = []
    for _, r in df.iterrows():
        name = str(r[fcol]); key = name.lower()
        quote, location, status = parse_excerpt_field(r[ecol])
        if status:
            rows.append((name, status, 0, 0, "", location, ""))
            continue
        path = idx.get(key) or idx.get(key.rsplit(".", 1)[0])
        if path is None:
            rows.append((name, "pdf_missing", 0, 0, "", location, ""))
            continue
        if path not in cache:
            cache[path] = load_pdf_text_and_pages(path)
        text, pages = cache[path]
        *result, matched = check_excerpt_with_pages(text, quote, pages)
        rows.append((name, *result, location, matched))
    out = pd.DataFrame(rows, columns=["filename", "status", "n_segments", "n_verified",
                                      "first_unverified_segment", "location", "matched_pages"])
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    vc = out.status.value_counts().to_dict()
    checkable = out[out.status.isin(["pass", "partial", "fail"])]
    n = len(checkable); k_fail = int((checkable.status == "fail").sum()); k_bad = int((checkable.status != "pass").sum())
    print("status counts:", vc)
    verified = out[out.status.isin(["pass", "pass_fragmented"])]
    print(f"verified excerpts with matched pages: {verified.matched_pages.ne('').sum()}/{len(verified)}")
    if vc.get("pass_fragmented"):
        print(f"fragmented matches (reported separately): {vc['pass_fragmented']}")
    if n:
        lo, hi = wilson(k_fail, n); lo2, hi2 = wilson(k_bad, n)
        print(f"checked {n} papers with a verifiable excerpt")
        print(f"  no match in PDF text: {k_fail}/{n} = {k_fail/n:.1%}  (95% CI {lo:.1%} ~ {hi:.1%})")
        print(f"  not fully verified (partial or fail): {k_bad}/{n} = {k_bad/n:.1%}  (95% CI {lo2:.1%} ~ {hi2:.1%})")
    print("Note: for 'fail', inspect first_unverified_segment and the PDF text. "
          "Text conversion differences or a non-quote field may prevent a match.")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
