"""Shared helpers for evaluating SLR Assistant *extraction* output (coating material, preparation method, source excerpt)."""
import math, re, unicodedata, difflib
from pathlib import Path
import pandas as pd

# ---------- table helpers ----------
def norm_col(c):
    return re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_")

def read_table(path):
    p = str(path)
    df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_excel(p)
    df.columns = [norm_col(c) for c in df.columns]   # "Coating Material" and coating_material both become coating_material
    return df

def pick_col(df, wanted, hints, label):
    if wanted and norm_col(wanted) in df.columns:
        return norm_col(wanted)
    for h in hints:
        c = [x for x in df.columns if h in x]
        if len(c) == 1:
            return c[0]
    raise SystemExit(f"Cannot find the {label} column. Columns present: {list(df.columns)}. Pass it explicitly with the matching --col-... option.")

def is_missing(x):
    s = "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x).strip().lower()
    return s in {"", "nan", "none", "not reported", "not_reported", "n/a", "na", "未报告", "未说明", "未提及", "无"}

def wilson(k, n, z=1.96):
    """95% Wilson interval; behaves sensibly when k=0 or k=n (unlike the bootstrap)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))

# ---------- coating material ----------
SYN = {"二氧化硅": "sio2", "氧化硅": "sio2", "氧化铝": "al2o3", "alumina": "al2o3", "氧化钛": "tio2", "二氧化钛": "tio2",
       "titania": "tio2", "氧化锆": "zro2", "zirconia": "zro2", "氧化镁": "mgo", "氧化钇": "y2o3", "氧化铬": "cr2o3",
       "氧化钙": "cao", "氮化钛": "tin", "氮化铬": "crn", "氮化钼": "mon", "氮化硅": "si3n4", "氮化铝": "aln",
       "氮化硼": "bn", "碳化钨": "wc", "碳化铬": "cr3c2", "碳化钛": "tic", "碳化硅": "sic", "碳化硼": "b4c",
       "石墨烯": "graphene", "金刚石类碳": "dlc", "类金刚石": "dlc"}
STOP = {"涂层", "镀层", "基层", "层", "膜", "薄膜", "coating", "coatings", "film", "films", "layer", "composite", "复合", "基",
        "掺杂", "掺", "添加", "doped", "doping", "with", "and", "和", "及", "为", "的", "含", "based", "base", "陶瓷",
        "ceramic", "wt", "vol", "at", "mol", "the", "of", "on"}

def material_components(text):
    """Turn a free-text material description into a set of normalised component tokens (formulas lower-cased)."""
    if is_missing(text):
        return set()
    s = unicodedata.normalize("NFKC", str(text)).lower()
    for zh in sorted(SYN, key=len, reverse=True):
        s = s.replace(zh, f" {SYN[zh]} ")
    s = re.sub(r"\d+(\.\d+)?\s*(wt|vol|at|mol)?\s*\.?\s*%", " ", s)      # drop contents such as 13wt%, 20%
    toks = [t for t in re.split(r"[^a-z0-9\u4e00-\u9fff]+", s) if t]
    out = set()
    for t in toks:
        if _CJK.search(t):                       # strip filler words glued to Chinese terms, e.g. 氧化镁基层 -> 氧化镁
            for sw in sorted((w for w in STOP if _CJK.search(w)), key=len, reverse=True):
                t = t.replace(sw, "")
        if t and t not in STOP and not re.fullmatch(r"\d+(\.\d+)?", t):
            out.add(t)
    return out

def score_material(ai, human):
    a, h = material_components(ai), material_components(human)
    inter, union = a & h, a | h
    jac = len(inter) / len(union) if union else 1.0
    if a == h:
        return "exact", jac, a, h
    return ("partial" if inter else "none"), jac, a, h

# ---------- preparation method ----------
# (category, patterns, generic).  A "generic" category is only used when no specific category matched.
METHOD_RULES = [
    ("微弧氧化/等离子电解氧化", [r"微弧氧化", r"等离子电解氧化", r"micro[- ]?arc oxidation", r"plasma electrolytic oxidation", r"\bmao\b", r"\bpeo\b"], False),
    ("超音速火焰喷涂(HVOF/HVAF)", [r"hvof", r"hvaf", r"超音速火焰", r"高速氧燃料", r"高速空气燃料", r"high[- ]velocity (oxy|air)"], False),
    ("大气等离子喷涂(APS)", [r"\baps\b", r"大气等离子", r"atmospheric plasma", r"等离子喷涂", r"plasma[- ]spray"], False),
    ("冷喷涂", [r"冷喷涂", r"cold[- ]spray"], False),
    ("磁控溅射", [r"磁控溅射", r"magnetron sputter", r"溅射", r"sputter"], False),
    ("阴极电弧/多弧离子镀", [r"多弧离子镀", r"阴极电弧", r"电弧离子镀", r"cathodic arc", r"arc ion plating", r"cathodic vacuum arc"], False),
    ("激光熔覆", [r"激光熔覆", r"激光.{0,6}熔覆", r"laser[- ]clad"], False),
    ("溶胶-凝胶", [r"溶胶", r"sol[- ]gel"], False),
    ("化学气相沉积(CVD)", [r"化学气相沉积", r"\bcvd\b", r"\bpecvd\b"], False),
    ("电沉积/化学镀", [r"电沉积", r"化学镀", r"电镀", r"electro[- ]?deposit", r"electroless"], False),
    ("阳极氧化", [r"阳极氧化", r"anodi[sz]"], False),
    ("其他热喷涂", [r"电弧喷涂", r"arc[- ]spray", r"火焰喷涂", r"flame[- ]spray", r"热喷涂", r"thermal[- ]spray"], True),
]
METHOD_CHOICES = [c for c, _, _ in METHOD_RULES] + ["复合工艺", "其他", "未说明"]

def classify_method(text):
    """Return (category, matched_categories). 'unrecognized' text maps to 其他 and should be reviewed by a person."""
    if is_missing(text):
        return "未说明", []
    s = unicodedata.normalize("NFKC", str(text)).lower()
    hits = [c for c, pats, g in METHOD_RULES if any(re.search(p, s) for p in pats)]
    specific = [c for c, pats, g in METHOD_RULES if c in hits and not g]
    hits = specific if specific else hits
    if not hits:
        return "其他", []
    return (hits[0] if len(hits) == 1 else "复合工艺"), hits

# ---------- source excerpt check ----------
_CJK = re.compile(r"[\u4e00-\u9fff]")

def norm_search(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    s = re.sub(r"-\s*\n\s*", "", s)               # hyphenation at line breaks
    s = re.sub(r"[*_`#>|\\\[\]]", " ", s)         # markdown produced by the PDF-to-text step
    return re.sub(r"\s+", "", s)

def split_excerpt(excerpt, min_len=12):
    parts = re.split(r"[\n;；。]|\.\.\.|…|\u2026", str(excerpt))
    segs = [norm_search(p) for p in parts]
    return [p for p in segs if len(p) >= min_len]

def check_excerpt(text_norm, excerpt, near=0.8):
    """Return (status, n_segments, n_verified, first_unverified_segment).
    status: pass | partial | fail | language_mismatch | no_excerpt | too_short"""
    if is_missing(excerpt):
        return "no_excerpt", 0, 0, ""
    segs = split_excerpt(excerpt)
    if not segs:
        return "too_short", 0, 0, ""
    ex_cjk = len(_CJK.findall(str(excerpt))) / max(len(str(excerpt)), 1)
    tx_cjk = len(_CJK.findall(text_norm)) / max(len(text_norm), 1)
    ok, first_bad = 0, ""
    for seg in segs:
        if seg in text_norm:
            ok += 1; continue
        sm = difflib.SequenceMatcher(None, text_norm, seg, autojunk=False)
        m = sm.find_longest_match(0, len(text_norm), 0, len(seg))
        if m.size / len(seg) >= near:
            ok += 1
        elif not first_bad:
            first_bad = seg[:120]
    if ok == len(segs):
        return "pass", len(segs), ok, ""
    if ok == 0 and ((ex_cjk > 0.3 and tx_cjk < 0.05) or (ex_cjk < 0.05 and tx_cjk > 0.3)):
        return "language_mismatch", len(segs), ok, first_bad   # model probably translated instead of quoting
    return ("partial" if ok else "fail"), len(segs), ok, first_bad

def load_pdf_text(path):
    p = Path(path)
    if p.suffix.lower() == ".txt":
        return p.read_text(encoding="utf-8", errors="ignore")
    try:
        import pymupdf4llm                      # same converter SLR uses, so the text matches what the model saw
        t = pymupdf4llm.to_markdown(str(p))
        if t.strip():
            return t
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            t = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        if t.strip():
            return t
    except Exception:
        pass
    return ""

def index_pdfs(folder):
    idx = {}
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in {".pdf", ".txt"}:
            idx.setdefault(p.name.lower(), p)
            idx.setdefault(p.stem.lower(), p)
    return idx
