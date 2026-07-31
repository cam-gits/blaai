import json
import re
from typing import List, Dict, Optional
from selectolax.parser import HTMLParser

IN_PATH = "data/raw/pages.jsonl"
OUT_PATH = "data/raw/chunks.jsonl"

THRESHOLD = 120  #min chunk length
CONTAINER_SELECTORS = ["main", "article", "div.entry-content", "div.content-area", "#content"]

STRIP_SELECTORS = ["script", "style", "noscript", "svg", "nav", "header", "footer", "form", "div.vitamin-credits"]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


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
    pages = 0
    discarded = 0
    total_chunks = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for obj in load_pages(input_path):
            pages += 1
            chunks = process_page(obj)
            if not chunks:
                discarded += 1
                continue
            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"read {pages} pages from {input_path}")
    print(f"discarded {discarded} pages below the {THRESHOLD}-char gate")
    print(f"wrote {total_chunks} chunks to {output_path}")


def main() -> None:
    chonk(IN_PATH, OUT_PATH)


if __name__ == "__main__":
    main()
