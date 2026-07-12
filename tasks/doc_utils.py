"""Download and cache Project Gutenberg texts; split into paragraphs."""
import os
import re
import urllib.request

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

BOOKS = {
    "alice": ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland"),
    "sherlock": ("https://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes"),
    "pride": ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice"),
}


def get_book(key):
    """Return the cleaned text of a book, downloading and caching on first use."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    path = os.path.join(DOCS_DIR, f"{key}.txt")
    if not os.path.exists(path):
        url, _ = BOOKS[key]
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8-sig", errors="replace")
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # strip Gutenberg header/footer (on read, so pre-downloaded files work too)
    m = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw)
    if m:
        raw = raw[m.end():]
    m = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG", raw)
    if m:
        raw = raw[: m.start()]
    return raw.strip()


def paragraphs(key, min_words=30, max_words=200):
    """Book paragraphs (single-spaced), filtered to a reasonable length."""
    text = get_book(key)
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if min_words <= len(p.split()) <= max_words]


def proper_nouns(text, top=5):
    """Most frequent capitalized non-sentence-initial words (crude NER)."""
    words = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", text)
    from collections import Counter

    common = {"The", "And", "But", "She", "He", "They", "There", "This", "That",
              "Then", "When", "What", "You", "His", "Her", "Not", "For", "With"}
    counts = Counter(w for w in words if w not in common)
    return [w for w, _ in counts.most_common(top)]
