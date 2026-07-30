import json
import time
import urllib.robotparser
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag, parse_qsl, urlencode, urlunparse

import httpx
from bs4 import BeautifulSoup   

SEED_URLS = [
    "https://www.waterfordcouncil.ie/"
]

ALLOWED_PREFIXES = [
    "waterfordcouncil.ie/"
]

OUT_PATH = Path("data/raw/pages.jsonl")

USER_AGENT = "BlaaBot/0.1 (civic information indexer)"
REQUEST_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 20.0
MAX_PAGES = 3000
MAX_DEPTH = 6
WANT_CONTENT_TYPE = "text/html"

def canonicalise(url: str) -> str:
    url, _frag = urldefrag(url)
    parts = urlparse(url)

    # Drop noisy query params
    DROP_PARAMS = {"print", "utm_source", "utm_medium", "utm_campaign", "sessionid"}
    kept = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in DROP_PARAMS]
    query = urlencode(kept)

    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((parts.scheme, parts.netloc, path, parts.params, query, ""))


def in_scope(url: str) -> bool:
    parts = urlparse(url)
    host = parts.netloc.removeprefix("www.")
    key = f"{host}{parts.path}"
    return any(key.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def extract_links(base_url: str, html: str) -> list[str]:
    """Find <a href> links, resolve to URLs"""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(base_url, a["href"])
        if absolute.startswith(("http://", "https://")):
            out.append(canonicalise(absolute))
    return out


_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

def robots_allows(url: str, agent: str) -> bool:
    parts = urlparse(url)
    root = f"{parts.scheme}://{parts.netloc}"
    rp = _robots_cache.get(root)
    if rp is None:
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.set_url(f"{root}/robots.txt")
            rp.read()
        except Exception:
            rp = None  # unreadable -> allow, but cache the decision
        _robots_cache[root] = rp
    if rp is None:
        return True
    return rp.can_fetch(agent, url)


def load_already_fetched(path: Path) -> set[str]:
    """Read back any existing JSONL so a re-run skips what we already have."""
    seen = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["url"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return seen


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    already = load_already_fetched(OUT_PATH)
    visited = set(already)                       # don't re-fetch or re-queue these
    frontier = deque()                           # (url, depth)

    for seed in SEED_URLS:
        c = canonicalise(seed)
        if c not in visited:
            frontier.append((c, 0))
            visited.add(c)

    headers = {"User-Agent": USER_AGENT}
    fetched_count = 0

    # Append mode: resumable across runs.
    out = OUT_PATH.open("a", encoding="utf-8")

    with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        while frontier and fetched_count < MAX_PAGES:
            url, depth = frontier.popleft()

            if not in_scope(url):
                continue
            if not robots_allows(url, USER_AGENT):
                print(f"robots-blocked: {url}")
                continue

            print(f"[{fetched_count+1}] depth {depth}: {url}")

            try:
                resp = client.get(url)
            except Exception as e:
                # Record the failure as a line too, so the manifest is complete.
                out.write(json.dumps({
                    "url": url, "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "status": f"error:{e.__class__.__name__}", "content_type": "", "html": "",
                }) + "\n")
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            content_type = resp.headers.get("content-type", "")
            record = {
                "url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "status": str(resp.status_code),
                "content_type": content_type,
                "html": "",
            }

            # Only store body + follow links for successful HTML responses.
            is_html = WANT_CONTENT_TYPE in content_type
            if resp.status_code == 200 and is_html:
                record["html"] = resp.text     # RAW markup, unparsed
                fetched_count += 1

                if depth < MAX_DEPTH:
                    for link in extract_links(url, resp.text):
                        if link not in visited and in_scope(link):
                            visited.add(link)
                            frontier.append((link, depth + 1))

            out.write(json.dumps(record) + "\n")
            time.sleep(REQUEST_DELAY_SECONDS)

    out.close()
    print(f"\nDone. Fetched {fetched_count} new pages this run.")
    print(f"Raw JSONL: {OUT_PATH}  (frontier had {len(frontier)} URLs left)")


if __name__ == "__main__":
    main()