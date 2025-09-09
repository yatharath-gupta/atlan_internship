import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json

visited = set()
docs = []

def extract_content(soup):
    """Extract structured content with headings and paragraphs."""
    content_parts = []
    
    # target the main content area if present
    main = soup.find("main") or soup.body
    
    for tag in main.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        if tag.name in ["h1", "h2", "h3"]:
            content_parts.append(f"\n## {text}\n")
        elif tag.name == "li":
            content_parts.append(f"- {text}")
        else:  # paragraph
            content_parts.append(text)
    
    return "\n".join(content_parts)


def scrape(url, domain=None, depth=0, max_depth=4):
    if depth > max_depth or url in visited:
        return
    visited.add(url)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string.strip() if soup.title else url
    content = extract_content(soup)

    docs.append({
        "url": url,
        "title": title,
        "content": content
    })
    print(f"[+] Scraped {url} (depth {depth})")

    # explore links
    for link in soup.find_all("a", href=True):
        full_url = urljoin(url, link["href"])
        parsed = urlparse(full_url)

        if domain is None:
            domain = parsed.netloc
        if parsed.netloc != domain:
            continue
        if parsed.scheme not in ("http", "https"):
            continue

        scrape(full_url, domain, depth + 1, max_depth)


def save_jsonl(path="docs_corpus25.jsonl"):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(docs)} docs into {path}")


if __name__ == "__main__":
    start_url = "https://docs.atlan.com/"
    scrape(start_url, max_depth=25)
    save_jsonl()
