"""Build the bounded Cochrane clinical-trial extraction benchmark."""

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pymupdf
import pymupdf4llm
import requests
from lxml import etree
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "test_output" / "medical_eval_v1"
OPENALEX = "https://api.openalex.org/works"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"
REVIEW_FILTER = (
    "primary_location.source.issn:1469-493X,has_pmcid:true,"
    "best_oa_location.license:cc-by|cc-by-nc"
)
# Prioritize intervention reviews with enough recent studies to yield open PDFs.
TOPICS = {
    "PMC11822883": "childhood myopia",
    "PMC12599494": "smoking cessation",
    "PMC10233616": "diabetes care",
    "PMC12178777": "stroke rehabilitation",
    "PMC10966941": "cardiovascular medication adherence",
    "PMC6367893": "primary care staffing",
    "PMC13539583": "preterm neuroprotection",
    "PMC9661877": "childhood eczema and food allergy",
    "PMC9832339": "breast cancer psychological care",
    "PMC8916090": "eczema treatment",
    "PMC8795871": "weight management",
}
TARGET = 20
REVIEW_LIMIT = 8
PER_REVIEW = 8
FIELD_NAMES = {
    "methods": "gold_study_design",
    "participants": "gold_population_and_sample_size",
    "interventions": "gold_intervention_and_comparator",
    "outcomes": "gold_outcomes_and_main_findings",
}
COLUMNS = [
    "paper_id", "pmcid", "doi", "license", "source_review_pmcid",
    "source_review_title", "study_id", *FIELD_NAMES.values(), "pdf_filename",
    "pdf_chars", "pdf_pages",
]
RANDOMIZED = re.compile(r"\brandomi[sz]|\brct\b|\brandom(?:ly)? allocat", re.I)
SECONDARY = re.compile(
    r"\b(?:protocol|secondary analys[ei]s|post.hoc|subgroup analysis|"
    r"systematic review|meta.analysis|economic evaluation|"
    r"(?:long.term|\d+[- ]year) follow.up|follow.up study|pilot protocol|"
    r"study design and rationale|baseline characteristics|cost.effectiveness analysis)\b", re.I
)
START = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?(?:introduction|background|"
                   r"methods?(?: and materials)?|materials and methods|patients and methods|"
                   r"trial design|study design|methodology)\s*$", re.I)
END = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?(?:references(?: and notes)?|"
                 r"bibliography|literature cited|works cited)\s*$", re.I)
S3_VERSION = re.compile(r"^(PMC\d+\.(\d+))/\1\.(xml|pdf|json)$")
ERRORS = (requests.RequestException, etree.XMLSyntaxError, ValueError, KeyError,
          OSError, RuntimeError)


def session():
    http = requests.Session()
    retry = Retry(total=3, backoff_factor=1, backoff_max=12,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods={"GET"}, respect_retry_after_header=False)
    http.mount("https://", HTTPAdapter(max_retries=retry))
    return http


def fetch(http, url, params=None):
    response = http.get(url, params=params, timeout=25)
    response.raise_for_status()
    return response


def xml_root(data):
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    return etree.fromstring(data, parser=parser)


def atomic_write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def pmcid(value):
    match = re.search(r"(?:^|/)(?:PMC)?(\d+)/?$", value or "", re.I)
    return f"PMC{match[1]}" if match else ""


def doi(value):
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value or "", flags=re.I).strip().casefold()


def pmid(value):
    return (value or "").strip("/").rsplit("/", 1)[-1]


def works(http, filter_value, fields, **params):
    return fetch(http, OPENALEX, {
        "filter": filter_value, "select": fields, "per_page": 100, **params,
    }).json()["results"]


def review_candidates(http):
    ids = "|".join(pid.removeprefix("PMC") for pid in TOPICS)
    results = works(http, f"{REVIEW_FILTER},pmcid:{ids}", "id,ids,doi,display_name")
    by_id = {pmcid(work["ids"].get("pmcid")): work for work in results}
    return [(pid, topic, by_id.get(pid)) for pid, topic in TOPICS.items()]


def versions(http, pid):
    params = {"list-type": 2, "prefix": f"{pid}."}
    files = {}
    while True:
        listing = xml_root(fetch(http, BUCKET, params).content)
        for key in listing.findall("{*}Contents/{*}Key"):
            match = S3_VERSION.fullmatch(key.text or "")
            if match:
                files.setdefault(int(match[2]), {})[match[3]] = key.text
        continuation = listing.findtext("{*}NextContinuationToken")
        if not continuation:
            return files
        params["continuation-token"] = continuation


def authorized_versions(http, files, extension):
    eligible = []
    for version, keys in sorted(files.items(), reverse=True):
        if not {"json", extension} <= keys.keys():
            continue
        meta = fetch(http, BUCKET + keys["json"]).json()
        license_code = (meta.get("license_code") or "").upper()
        if (meta.get("is_pmc_openaccess") and license_code
                and license_code not in {"TDM", "NO-CC CODE", "UNKNOWN"}
                and not meta.get("is_retracted") and not meta.get("is_historical_ocr")):
            eligible.append((version, keys, meta))
    return sorted(eligible, key=lambda item: bool(item[2].get("is_manuscript")))


def article_file(http, keys, meta, extension):
    url = meta.get(f"{extension}_url") or ""
    link = urlsplit(url)
    if link.netloc != "pmc-oa-opendata" or link.path.lstrip("/") != keys[extension]:
        raise ValueError(f"PMC metadata does not identify the listed {extension} object")
    payload = fetch(http, BUCKET + keys[extension]).content
    expected = parse_qs(link.query).get("md5", [""])[0]
    if expected and hashlib.md5(payload).hexdigest() != expected:
        raise ValueError(f"PMC {extension} checksum mismatch")
    return payload


def jats_text(element):
    if element is None:
        return ""

    def content(node):
        yield node.text or ""
        for child in node:
            block = child.tag in {"break", "p", "list-item", "list"}
            if block:
                yield "\n"
            if isinstance(child.tag, str):
                yield from content(child)
            if block:
                yield "\n"
            yield child.tail or ""

    return "\n".join(line for raw in "".join(content(element)).splitlines()
                     if (line := " ".join(raw.split())))


def study_label(label):
    return " ".join(re.findall(r"\w+", label.split("{", 1)[0].casefold()))


def references(nodes):
    for ref in nodes:
        citation = ref.find("mixed-citation")
        if citation is None:
            citation = ref.find("element-citation")
        if citation is None or citation.get("publication-type") not in {None, "journal"}:
            continue
        title = jats_text(citation.find("article-title"))
        ids = {}
        for entry in citation.iter("pub-id"):
            ids[entry.get("pub-id-type")] = jats_text(entry)
        if title and not SECONDARY.search(title) and (ids.get("doi") or ids.get("pmid")):
            yield {
                "id": ref.get("id"), "title": title, "year": citation.findtext("year", ""),
                "author": jats_text(citation.find(".//surname")),
                "doi": doi(ids.get("doi")), "pmid": pmid(ids.get("pmid")),
                "pmcid": pmcid(ids.get("pmcid")),
            }


def parse_studies(root):
    sections = [node for node in root.iter("sec")
                if jats_text(node.find("title")).casefold().startswith(
                    "characteristics of included studies")]
    tables = {table.get("id") or table: table for section in sections
              for table in section.iter("table-wrap")}
    ref_groups = {}
    ref_labels = {}
    for section in root.iter("ref-list"):
        if not jats_text(section.find("title")).casefold().startswith(
                "references to studies included"):
            continue
        for group in section.iter("ref-list"):
            ref_groups[group.get("id")] = group.findall("ref")
            label = study_label(jats_text(group.find("title")))
            if label:
                ref_labels.setdefault(label, []).append(group.findall("ref"))
    seen = set()
    studies = []
    for table in tables.values():
        caption = table.find("caption")
        label = jats_text(caption)
        key = study_label(label)
        gold = {}
        for tr in table.findall(".//tr"):
            cells = tr.xpath("./td|./th")
            if len(cells) >= 2:
                field = FIELD_NAMES.get(jats_text(cells[0]).strip(":").casefold())
                val = jats_text(cells[1])
                if field and val:
                    gold[field] = "\n".join(filter(None, (gold.get(field), val)))
        if (not key or key in seen or len(gold) != len(FIELD_NAMES)
                or not RANDOMIZED.search(gold["gold_study_design"])):
            continue
        seen.add(key)
        links = [node.get("rid") for node in caption.iter("xref")
                 if node.get("ref-type") == "ref-list"]
        linked = [ref_groups[rid] for rid in links if rid in ref_groups]
        groups = linked or ref_labels.get(key, [])
        refs = list(references(groups[0])) if len(groups) == 1 else []
        studies.append({
            "study_id": label, "gold": gold, "table_id": table.get("id"),
            "references": refs,
        })
    return len(tables), studies


def primary_refs(study):
    refs = study["references"]
    if len(refs) == 1:
        return refs
    label = study_label(study["study_id"])
    year = re.search(r"\b(?:19|20)\d{2}\b", label)
    if not year:
        return []
    exact = [ref for ref in refs if ref["year"] == year[0]
             and (author := study_label(ref["author"])) and label.startswith(author + " ")]
    if len(exact) == 1:
        return exact
    return []


def map_trials(http, refs):
    by_id = {}
    with_pmid = {ref["pmid"] for ref in refs if ref["pmid"]}
    with_doi = {ref["doi"] for ref in refs if ref["doi"] and not ref["pmid"]}
    for field, values in (("pmid", with_pmid), ("doi", with_doi)):
        for offset in range(0, len(values), 35):
            ids = sorted(values)[offset:offset + 35]
            for work in works(http, f"{field}:{'|'.join(ids)}", "id,ids,doi,display_name"):
                for kind, normalize in (("doi", doi), ("pmid", pmid)):
                    value = normalize(work["ids"].get(kind))
                    if value:
                        by_id.setdefault((kind, value), {})[work["id"]] = work
    return by_id


def resolve_trial(ref, index):
    candidates = {}
    for kind in ("pmid", "doi"):
        candidates.update(index.get((kind, ref[kind]), {}))
    matches = set()
    for work in candidates.values():
        ids = work["ids"]
        if ref["doi"] and ids.get("doi") and doi(ids["doi"]) != ref["doi"]:
            continue
        if ref["pmid"] and ids.get("pmid") and pmid(ids["pmid"]) != ref["pmid"]:
            continue
        if pid := pmcid(ids.get("pmcid")):
            matches.add(pid)
    if ref["pmcid"]:
        matches.add(ref["pmcid"])
    return next(iter(matches)) if len(matches) == 1 else ""


def pdf_stats(payload):
    if not payload.startswith(b"%PDF-"):
        raise ValueError("file does not have a PDF header")
    with pymupdf.open(stream=payload, filetype="pdf") as document:
        pages = len(document)
        native = [page.get_text() for page in document]
        if sum(sum(char.isalpha() for char in page) >= 100 for page in native) * 2 < pages:
            raise ValueError("most pages lack a native text layer")
        if sum(sum(char.isalpha() for char in page) for page in native) < 3000:
            raise ValueError("insufficient native text")
        markdown = pymupdf4llm.to_markdown(document, use_ocr=False, show_progress=False)
    lines = markdown.splitlines()
    headings = [line.strip().strip("#* _").strip() for line in lines]
    start = next((idx for idx, heading in enumerate(headings) if START.fullmatch(heading)), None)
    if start is None:
        raise ValueError("no recognizable main-text heading")
    end = next((idx for idx in range(start + 1, len(headings))
                if END.fullmatch(headings[idx])), len(lines))
    body = " ".join(" ".join(lines[start:end]).split())
    if len(body) < 5000:
        raise ValueError(f"main text below 5000 characters: {len(body)}")
    corrupt = sum(char == "\ufffd" or unicodedata.category(char) in {"Co", "Cc"} for char in body)
    if corrupt > 0.005 * len(body) or sum(char.isalpha() for char in body) < len(body) / 2:
        raise ValueError("main text has excessive non-text or invalid characters")
    return len(body), pages


def collect_pdf(http, pid, ref):
    errors = []
    for version, keys, meta in authorized_versions(http, versions(http, pid), "pdf"):
        if pmcid(meta.get("pmcid")) != pid:
            continue
        if ref["doi"] and meta.get("doi") and doi(meta["doi"]) != ref["doi"]:
            continue
        if ref["pmid"] and meta.get("pmid") and str(meta["pmid"]) != ref["pmid"]:
            continue
        if not (ref["doi"] and doi(meta.get("doi")) == ref["doi"]
                or ref["pmid"] and str(meta.get("pmid")) == ref["pmid"]):
            continue
        try:
            payload = article_file(http, keys, meta, "pdf")
            chars, pages = pdf_stats(payload)
        except ERRORS as exc:
            errors.append(f"{version}: {exc}")
            continue
        return payload, {
            "pdf_chars": chars, "pdf_pages": pages, "license": meta["license_code"],
            "doi": doi(meta.get("doi")) or ref["doi"], "pdf_version": version,
            "pdf_url": BUCKET + keys["pdf"], "pdf_metadata_url": BUCKET + keys["json"],
            "pdf_sha256": hashlib.sha256(payload).hexdigest(),
        }
    raise ValueError(" | ".join(errors) or "no authorized PDF with matching trial identity")


def collect_review(http, source, papers, used, used_dois):
    licensed = authorized_versions(http, versions(http, source["pmcid"]), "xml")
    for version, keys, meta in licensed:
        xml = article_file(http, keys, meta, "xml")
        count, studies = parse_studies(xml_root(xml))
        if not count:
            continue
        source.update(xml_url=BUCKET + keys["xml"], review_license=meta["license_code"],
                      tables=count, randomized_complete_tables=len(studies), parsed=True,
                      xml_sha256=hashlib.sha256(xml).hexdigest(), trials=[])
        atomic_write(OUTPUT / "cache" / f'{source["pmcid"]}.{version}.xml', xml)
        break
    else:
        source["status"] = "no_authorized_included_studies_xml"
        return
    for study in studies:
        study["primary"] = primary_refs(study)
    trial_index = map_trials(http, [ref for study in studies for ref in study["primary"]])
    for study in studies:
        if source["accepted"] >= PER_REVIEW or len(papers) >= TARGET:
            break
        audit = {"study_id": study["study_id"], "table_id": study["table_id"],
                 "references": study["references"]}
        source["trials"].append(audit)
        if not study["primary"]:
            audit["status"] = "primary_report_not_unambiguous"
            continue
        ref = study["primary"][0]
        pid = resolve_trial(ref, trial_index)
        audit.update(primary=ref, pmcid=pid)
        if not pid or pid in used or ref["doi"] and ref["doi"] in used_dois:
            audit["status"] = "no_unambiguous_pmcid" if not pid else "duplicate_trial_paper"
            continue
        try:
            payload, info = collect_pdf(http, pid, ref)
        except ERRORS as exc:
            audit.update(status="pdf_rejected", reason=str(exc))
            continue
        if info["doi"] and info["doi"] in used_dois:
            audit["status"] = "duplicate_trial_paper"
            continue
        filename = f"{pid}.pdf"
        atomic_write(OUTPUT / "pdfs" / filename, payload)
        paper = {
            "paper_id": f"P{len(papers) + 1:03d}", "pmcid": pid,
            "source_review_pmcid": source["pmcid"], "source_review_title": source["title"],
            "study_id": study["study_id"], "pdf_filename": filename,
            **study["gold"], **info,
        }
        papers.append(paper)
        used.add(pid)
        if info["doi"]:
            used_dois.add(info["doi"])
        source["accepted"] += 1
        audit.update(status="accepted", paper_id=paper["paper_id"])
        print(f'{paper["paper_id"]} {pid} {source["topic"]} {study["study_id"]}', flush=True)
    source["status"] = "processed"


def write_outputs(sources, papers, result):
    topics = sorted({s["topic"] for s in sources if s["accepted"]})
    for source in sources:
        trials = source.get("trials", [])
        source["studies_examined"] = len(trials)
        source["pmcid_matches"] = sum(bool(row.get("pmcid")) for row in trials)
    manifest = {
        "status": result, "target_papers": TARGET, "accepted_papers": len(papers),
        "review_limit": REVIEW_LIMIT, "per_review_limit": PER_REVIEW,
        "reviews_scanned": sum(bool(s.get("parsed")) for s in sources),
        "topics": topics, "openalex_filter": REVIEW_FILTER, "reviews": sources,
        "papers": [{key: val for key, val in row.items() if key not in FIELD_NAMES.values()}
                   for row in papers],
    }
    csv_path = OUTPUT / "gold_standard.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(papers)
    temporary.replace(csv_path)
    atomic_write(OUTPUT / "manifest.json",
                 (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    return manifest


def main():
    for directory in ("pdfs", "cache"):
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)
    sources, papers, used, used_dois = [], [], set(), set()
    with session() as http:
        for pid, topic, work in review_candidates(http):
            if len(papers) >= TARGET or sum(bool(s.get("parsed")) for s in sources) >= REVIEW_LIMIT:
                break
            source = {"pmcid": pid, "topic": topic, "accepted": 0}
            sources.append(source)
            if not work:
                source["status"] = "not_returned_by_openalex_license_filter"
                continue
            source.update(title=work["display_name"], openalex_id=work["id"], doi=work["doi"])
            try:
                collect_review(http, source, papers, used, used_dois)
            except ERRORS as exc:
                source.update(status="source_error", reason=str(exc))
            print(f'{pid}: {source["status"]}, accepted={source["accepted"]}', flush=True)
    result = "complete" if len(papers) == TARGET else "insufficient_samples"
    manifest = write_outputs(sources, papers, result)
    print(f'{result}: {len(papers)}/{TARGET}, {len(manifest["topics"])} topics', flush=True)
    return 0 if result == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
