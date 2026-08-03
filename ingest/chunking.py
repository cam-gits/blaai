import json
import re
from typing import List, Dict, Optional
from selectolax.parser import HTMLParser
from collections import defaultdict
from urllib.parse import urlparse, urlunparse

IN_PATH = "data/raw/pages.jsonl"
OUT_PATH = "data/raw/chunks.jsonl"

THRESHOLD = 120  #min chunk length
BOILERPLATE_MIN_PAGES = 4  #max chonk frequency
SKIP_QUERY_FLAGS = ("preview=true",)
CONTAINER_SELECTORS = ["main", "article", "div.entry-content", "div.content-area", "#content"]

STRIP_SELECTORS = ["script", "style", "noscript", "svg", "nav", "header", "footer", "form", "div.vitamin-credits"]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _canonical_url(url: str) -> str:
    #collapse variants of same page
    p = urlparse(url)
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return urlunparse(("https", netloc, p.path.rstrip("/") or "/", "", "", ""))


def _page_frequency(records: List[Dict]) -> Dict[str, int]:
    #identify repeated chunks
    pages_per_text: Dict[str, set] = defaultdict(set)
    for r in records:
        pages_per_text[r["chunk"]].add(_canonical_url(r["URL"]))
    return {text: len(urls) for text, urls in pages_per_text.items()}


def _pick_container(tree: HTMLParser):
    for sel in CONTAINER_SELECTORS:
        node = tree.css_first(sel)
        if node is not None:
            return node
    return tree.body


def extract_blocks(html: str) -> List[Dict[str, str]]:
    tree = HTMLParser(html)
    for sel in STRIP_SELECTORS:
        for node in tree.css(sel):
            node.decompose()
 
    container = _pick_container(html_tree := tree) and _pick_container(tree)
    if container is None:
        return []
 
    blocks: List[Dict[str, str]] = []
    consumed = set()  
 
    for node in container.traverse(include_text=False):
        tag = node.tag
        nid = id(node)

        #header tag
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = _normalise(node.text(deep=True))
            if text:
                blocks.append({"type": "heading", "text": text})

        #append lists
        elif tag in ("ul", "ol"):
            items = []
            for li in node.css("li"):
                consumed.add(id(li))
                li_text = _normalise(li.text(deep=True))
                if li_text:
                    items.append(li_text)
            if items:
                blocks.append({"type": "body", "text": "; ".join(items)})

        #chonk
        elif tag == "p":
            if nid in consumed:
                continue
            text = _normalise(node.text(deep=True))
            if text:
                blocks.append({"type": "body", "text": text})
 
    return blocks


#is there meaningful textual data in the scraped page
def page_meaningful_chars(blocks: List[Dict[str, str]]) -> int:
    return sum(len(b["text"]) for b in blocks if b["type"] == "body")
 

def chunk_blocks(blocks: List[Dict[str, str]], url: str) -> List[Dict]:

    if page_meaningful_chars(blocks) < THRESHOLD:
        return []

    chunks: List[Dict] = []
    current_heading: Optional[str] = None
    lead_buffer = ""          #in case page starts on thin chunk
    lead_heading: Optional[str] = None  #h1

    for block in blocks:
        if block["type"] == "heading":
            current_heading = block["text"]
            continue

        text = block["text"]

        if len(text) >= THRESHOLD:
            #substantial data path
            if lead_buffer:
                #prepend buffer if avail
                text = lead_buffer + " " + text
                heading_for_chunk = lead_heading
                lead_buffer = ""
                lead_heading = None
            else:
                heading_for_chunk = current_heading
            chunks.append({
                "URL": url,
                "heading": heading_for_chunk,
                "chunk": text,
                "chunk_index": len(chunks),
            })
        else:
            #think path, append back if possible or hold to next
            if chunks:
                chunks[-1]["chunk"] += " " + text
            else:
                if not lead_buffer:
                    lead_heading = current_heading
                lead_buffer = (lead_buffer + " " + text).strip() if lead_buffer else text

    #in case of data left in buffer
    if lead_buffer:
        chunks.append({
            "URL": url,
            "heading": lead_heading,
            "chunk": lead_buffer,
            "chunk_index": len(chunks),
        })
 
    return chunks


def process_page(obj: Dict) -> List[Dict]:
    blocks = extract_blocks(obj["html"])
    return chunk_blocks(blocks, obj["url"])


def load_pages(input_path: str):
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{{"):
                line = line[1:]
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def chonk(input_path: str = IN_PATH, output_path: str = OUT_PATH) -> None:
    output_path: str = OUT_PATH) -> None:
    pages = 0
    discarded = 0
    skipped = 0

    records: List[Dict] = []
    for obj in load_pages(input_path):
        if any(flag in urlparse(obj["url"]).query for flag in SKIP_QUERY_FLAGS):
            skipped += 1
            continue
        pages += 1
        chunks = process_page(obj)
        if not chunks:
            discarded += 1
            continue
        records.extend(chunks)

    #second pass to remove excess dupes (low numbers = legitamately reinforced data, high = boilerplate)
    frequency = _page_frequency(records)
    per_page_index: Dict[str, int] = defaultdict(int)
    kept: List[Dict] = []
    dropped = 0

    for record in records:
        if frequency[record["chunk"]] >= BOILERPLATE_MIN_PAGES:
            dropped += 1
            continue
        record["chunk_index"] = per_page_index[record["URL"]]
        per_page_index[record["URL"]] += 1
        kept.append(record)

    with open(output_path, "w", encoding="utf-8") as out:
        for record in kept:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"read {pages} pages from {input_path}")
    print(f"skipped {skipped} preview/draft URLs")
    print(f"discarded {discarded} pages below the {THRESHOLD}-char gate")
    print(f"dropped {dropped} boilerplate chunks (text on >= {BOILERPLATE_MIN_PAGES} pages)")
    print(f"wrote {len(kept)} chunks to {output_path}")


def main() -> None:
    chonk(IN_PATH, OUT_PATH)


if __name__ == "__main__":
    main()
