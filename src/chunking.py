import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config
from src.abstract_parser import parse_pubmed_file


def create_documents(abstracts_dir, mesh_lookup):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
    )

    documents = []
    ids = []
    skipped = []

    for filename in sorted(os.listdir(abstracts_dir)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(abstracts_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        parsed = parse_pubmed_file(raw_text)
        pubid = parsed["pmid"] if parsed["pmid"] else filename.replace(".txt", "")

        if not parsed["sections"] or all(
            not s["text"].strip() for s in parsed["sections"]
        ):
            skipped.append(filename)
            continue

        meshes = mesh_lookup.get(str(pubid), [])

        for sec_idx, section in enumerate(parsed["sections"]):
            label = section["label"]
            text = section["text"]
            chunks = text_splitter.split_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                page_content = f"[{label}] {chunk}"
                if sec_idx == 0 and chunk_idx == 0:
                    page_content = f"Title: {parsed['title']}\n{page_content}"

                doc_id = f"{pubid}_sec{sec_idx}_chunk{chunk_idx}"
                document = Document(
                    page_content=page_content,
                    metadata={
                        "pubid": str(pubid),
                        "title": parsed["title"],
                        "authors": parsed["authors"],
                        "doi": parsed["doi"],
                        "journal": parsed["journal"],
                        "publication_year": parsed["publication_year"],
                        "meshes": ",".join(meshes),
                        "abstract_section": label,
                        "section_index": sec_idx,
                        "chunk_index": chunk_idx,
                        "total_chunks_in_section": len(chunks),
                        "dataset": "pubmed_abstract",
                    },
                    id=doc_id,
                )
                documents.append(document)
                ids.append(doc_id)

    unique_abstracts = len(set(d.metadata["pubid"] for d in documents))
    print(f"Parsed {len(documents)} chunks from {unique_abstracts} abstracts")
    if skipped:
        print(f"Skipped {len(skipped)} abstracts with no content: {skipped[:10]}...")

    return documents, ids
