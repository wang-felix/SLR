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
MATERIAL_ALIASES = {
    "二氧化硅": "SiO2", "氧化硅": "SiO2", "氧化铝": "Al2O3", "alumina": "Al2O3",
    "氧化钛": "TiO2", "二氧化钛": "TiO2", "titania": "TiO2",
    "氧化锆": "ZrO2", "zirconia": "ZrO2", "氧化镁": "MgO", "氧化钇": "Y2O3",
    "氧化铬": "Cr2O3", "氧化钙": "CaO", "氮化钛": "TiN", "氮化铬": "CrN",
    "氮化钼": "MoN", "氮化硅": "Si3N4", "氮化铝": "AlN", "氮化硼": "BN",
    "碳化钨": "WC", "碳化铬": "Cr3C2", "碳化钛": "TiC", "碳化硅": "SiC",
    "碳化硼": "B4C", "石墨烯": "graphene", "金刚石类碳": "DLC", "类金刚石": "DLC",
}
_ELEMENT_SYMBOLS = set("""
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
    Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce
    Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At
    Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn
    Nh Fl Mc Lv Ts Og
""".split())
_COUNT = r"(?:\d+(?:\.\d+)?)?"
_ATOM = rf"[A-Z][a-z]?{_COUNT}"
_CHEMICAL_FORMULA = re.compile(rf"(?:{_ATOM}|\((?:{_ATOM})+\){_COUNT})+")
_ELEMENT_IN_FORMULA = re.compile(r"[A-Z][a-z]?")
_MATERIAL_TOKEN = re.compile(
    r"(?<![A-Za-z0-9.])[A-Za-z](?:[A-Za-z0-9]|\.(?=\d)|\([A-Za-z0-9.]+\)\d+)*"
    r"(?![A-Za-z0-9]|\.\d)"
)
_MATERIAL_ALIAS_PATTERN = re.compile(
    "|".join(
        rf"\b{word}\b" if word.isascii() else word
        for word in sorted(MATERIAL_ALIASES, key=len, reverse=True)
    ),
    re.IGNORECASE,
)
_KNOWN_MATERIALS = {value.casefold(): value for value in MATERIAL_ALIASES.values()}
_ELECTROLYTE_RECIPE = re.compile(r"\([^()]*\d\s*(?:g|mg|mol|mmol)\s*/\s*L\b[^()]*\)", re.IGNORECASE)
_MATERIAL_CLAUSE_END = (
    r"(?=\bcoatings?\s+(?:consists?|contains?|comprises?)\b|[;；。!?\n]|\.(?=\s|$)|$)"
)
_SUBSTRATE = re.compile(
    r"\b(?:(?<!based\s)on(?:to)?\s+|substrates?\s*[:=]\s*).*?"
    r"(?=\bwith\b|[,，]|" + _MATERIAL_CLAUSE_END + r")"
    r"|以[^,，;；。]+为(?:基材|衬底|基底)|在[^,，;；。]+(?:表面|上)(?=(?:原位)?(?:沉积|制备|形成))",
    re.IGNORECASE,
)
_ELECTROLYTE = re.compile(
    r"(?:\b(?:in|into|using)\s+(?:[^;；。()\n]*?\s)?electrolytes?\b"
    r"|\belectrolytes?\s*(?:[:=]|contains?\b|consists?\s+of\b)).*?"
    r"(?=\bwith\b[^;；。,]*\badditives?\b|" + _MATERIAL_CLAUSE_END + r")"
    r"|电解液(?:组分)?\s*(?:含有?|[:：])[^;；。\n]*",
    re.IGNORECASE,
)
_DOPANT_AMOUNT = re.compile(
    r"\b(?:different|various)\s+(?P<element>[A-Z][a-z]?)\s+doping\s+contents?\b"
    r"(?:\s*\([^()]*%[^()]*\))?"
)
_MEASUREMENT_UNIT = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?\s+(?:V|kV|Pa|kPa|MPa|W|A|K|N)\b")
_OXIDATION_STATE_LIST = re.compile(
    r"\boxides?\s+of\s+[^();]+?\bin\s+(?:complete\s+)?oxidation\s+states?\s*\([^()]*\)",
    re.IGNORECASE,
)
_ILLUSTRATIVE_MATERIAL = re.compile(r"\b(?:e\.g\.|for example\b|such as\b)|例如|诸如", re.IGNORECASE)
_LAYER_NOTATION = re.compile(r"\b(?:multilayer|duplex)\b|多层|双层", re.IGNORECASE)

def material_components(text):
    """Extract identifiable coating components from a list or a free-text answer."""
    if is_missing(text):
        return set()
    s = unicodedata.normalize("NFKC", str(text))
    s = _ELECTROLYTE_RECIPE.sub(" ", s)
    s = _ELECTROLYTE.sub(" ", s)
    s = _SUBSTRATE.sub(" ", s)
    varied_dopants = {match["element"] for match in _DOPANT_AMOUNT.finditer(s)}
    s = _DOPANT_AMOUNT.sub(" ", s)
    s = _MEASUREMENT_UNIT.sub(" ", s)
    s = _OXIDATION_STATE_LIST.sub(" oxides ", s)
    s = _MATERIAL_ALIAS_PATTERN.sub(lambda m: f" {MATERIAL_ALIASES[m[0].casefold()]} ", s)

    components = set()
    for match in _MATERIAL_TOKEN.finditer(s):
        token = match[0]
        if _CHEMICAL_FORMULA.fullmatch(token) and all(
            symbol in _ELEMENT_SYMBOLS for symbol in _ELEMENT_IN_FORMULA.findall(token)
        ):
            components.add(token)
        elif token.casefold() in _KNOWN_MATERIALS:
            components.add(_KNOWN_MATERIALS[token.casefold()])
    # A doping-level annotation can repeat an element already encoded in the compound.
    reported_elements = {
        symbol for formula in components for symbol in _ELEMENT_IN_FORMULA.findall(formula)
    }
    components.update((varied_dopants & _ELEMENT_SYMBOLS) - reported_elements)
    return components

def score_material(ai, human):
    a, h = material_components(ai), material_components(human)
    raw_answer = str(ai)
    if (not h or (not a and not is_missing(ai))
            or _ILLUSTRATIVE_MATERIAL.search(raw_answer)
            or ("/" in raw_answer and _LAYER_NOTATION.search(raw_answer))):
        return "review", None, a, h
    inter, union = a & h, a | h
    jac = len(inter) / len(union)
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
_LOCATION_PREFIX = re.compile(r"^(?:(?:section|table|fig(?:ure)?)\b|sec\.|§)", re.IGNORECASE)


def parse_excerpt_field(value):
    """Return (quote, location, status); status is None for text to check."""
    text = "" if is_missing(value) else str(value)
    quote, separator, location = text.partition(" || ")
    if separator:
        return quote.strip(), location.strip(), None
    text = text.strip()
    if text == "NOT_FOUND":
        return "", "", "not_found"
    if _LOCATION_PREFIX.match(text):
        return "", text, "location_only"
    return text, "", None

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
