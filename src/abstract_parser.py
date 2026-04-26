import re
from langdetect import detect, LangDetectException

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config

SECTION_PATTERN = "|".join(
    re.escape(label)
    for label in sorted(config.SECTION_LABELS, key=len, reverse=True)
)

SECTION_REGEX = re.compile(
    rf"^({SECTION_PATTERN})\s*:\s*", re.IGNORECASE | re.MULTILINE
)

FOOTER_REGEX = re.compile(
    r"^(DOI:|PMID:|PMCID:|Published by|Copyright|Erratum in|"
    r"Comment in|Retraction in|Update in)",
    re.MULTILINE,
)


def is_english(text, min_length=20):
    if len(text.strip()) < min_length:
        return True
    try:
        return detect(text) == "en"
    except LangDetectException:
        return True


def parse_pubmed_file(raw_text):
    text = raw_text.strip()

    pmid_match = re.search(r"PMID:\s*(\d+)", text)
    pmid = pmid_match.group(1) if pmid_match else None
    if not pmid:
        alt = re.search(r"\*\s*PMID:\s*(\d+)", text)
        pmid = alt.group(1) if alt else None

    doi_match = re.search(r"(?:DOI|doi):\s*(10\.\S+)", text)
    doi = doi_match.group(1).rstrip(".") if doi_match else ""

    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    journal = ""
    publication_year = ""
    if paragraphs:
        citation = " ".join(paragraphs[0].split())
        journal_match = re.match(r"\d+\.\s*(.+?)\.?\s+\d{4}", citation)
        if journal_match:
            journal = journal_match.group(1).strip().rstrip(".")
        year_match = re.search(r"\b(19|20)\d{2}\b", citation)
        if year_match:
            publication_year = year_match.group(0)

    title = " ".join(paragraphs[1].split()) if len(paragraphs) > 1 else ""

    authors = ""
    if len(paragraphs) > 2:
        candidate = paragraphs[2]
        if re.search(r"\(\d+\)", candidate):
            authors = " ".join(candidate.split())

    matches = list(SECTION_REGEX.finditer(text))

    if matches:
        abstract_start = matches[0].start()
        footer_match = FOOTER_REGEX.search(text, pos=abstract_start)
        abstract_end = footer_match.start() if footer_match else len(text)
        abstract_text = text[abstract_start:abstract_end].strip()

        section_matches = list(SECTION_REGEX.finditer(abstract_text))
        sections = []
        for i, match in enumerate(section_matches):
            label = match.group(1).upper()
            start = match.end()
            end = (
                section_matches[i + 1].start()
                if i + 1 < len(section_matches)
                else len(abstract_text)
            )
            section_text = " ".join(abstract_text[start:end].split())
            if section_text and is_english(section_text):
                sections.append({"label": label, "text": section_text})
    else:
        abstract_text = ""
        for i, para in enumerate(paragraphs):
            if para.strip().lower() == "abstract":
                remaining = "\n\n".join(paragraphs[i + 1 :])
                footer_match = FOOTER_REGEX.search(remaining)
                abstract_text = (
                    remaining[: footer_match.start()].strip()
                    if footer_match
                    else remaining.strip()
                )
                break

        if not abstract_text and len(paragraphs) > 4:
            body_parts = []
            for para in paragraphs[4:]:
                if FOOTER_REGEX.match(para):
                    break
                body_parts.append(para)
            abstract_text = " ".join(" ".join(body_parts).split())

        if abstract_text and is_english(abstract_text):
            sections = [{"label": "ABSTRACT", "text": abstract_text}]
        else:
            sections = []

    return {
        "pmid": pmid,
        "title": title,
        "authors": authors,
        "doi": doi,
        "journal": journal,
        "publication_year": publication_year,
        "sections": sections,
    }
