'''
Provides web browsing tools for the assistant
- Web search via SearXNG with DuckDuckGo fallback
- Page content fetching via BeautifulSoup for static pages, Playwright for dynamic
- URL opening in the default browser
- Browser tab management via Playwright
'''

import os
import re
import webbrowser
import requests
from bs4 import BeautifulSoup
from config import DEBUG
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, parse_qs, unquote

load_dotenv()

# SearXNG instance URL — set in .env
# Falls back to DuckDuckGo if not configured
SEARXNG_URL = os.getenv("SEARXNG_URL", "")

def search_web(query, num_results=5):
    """
    Search the web using SearXNG if avaliable, otherwise DuckDuckGo
    Returns formatted search results as text for the model to read

    query: search query string
    num_results: how many results to return
    returns: formatted string of search results
    """
    if SEARXNG_URL:
        return search_searxng(query, num_results)
    else:
        return search_duckduckgo(query, num_results)


def search_searxng(query, num_results=5):
    """
    Search via self-hosted SearXNG instance
    
    query: search query string
    num_results: how many results to return
    returns: formatted results string
    """
    try:
        if DEBUG:
            print(f"[BROWSER] Trying SearXNG at: {SEARXNG_URL}")
        
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general"
            },
            timeout=10
        )
        
        if DEBUG:
            print(f"[BROWSER] SearXNG status: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        
        if DEBUG:
            print(f"[BROWSER] SearXNG results count: {len(data.get('results', []))}")

        results = data.get("results", [])[:num_results]
        if not results:
            return f"[INFO] No results found for '{query}'"

        lines = [f"[Search results for '{query}']"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.get('title', 'No title')}")
            lines.append(f"   URL: {r.get('url', '')}")
            lines.append(f"   {r.get('content', 'No description')[:200]}")

        return "\n".join(lines)

    except requests.exceptions.ConnectionError:
        if DEBUG:
            print(f"[BROWSER] SearXNG unavailable, falling back to DuckDuckGo")
        return search_duckduckgo(query, num_results)
    except Exception as e:
        return f"[ERROR] Search failed: {e}"


def search_duckduckgo(query, num_results=5):
    """
    Search via DuckDuckGo instant answers API if searxng not avaliable

    query: search query string
    num_results: how many results to return
    returns: formatted results string
    """
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            },
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        data = response.json()

        lines = [f"[Search results for '{query}' via DuckDuckGo]"]

        # instant answer if available
        if data.get("AbstractText"):
            lines.append(f"Source: {clean_duckduckgo_url(data['AbstractURL'])}")
            if data.get("AbstractURL"):
                lines.append(f"Source: {data['AbstractURL']}")

        # related topics
        topics = data.get("RelatedTopics", [])[:num_results]
        if topics:
            lines.append("\nRelated results:")
            for i, topic in enumerate(topics, 1):
                if isinstance(topic, dict) and topic.get("Text"):
                    lines.append(f"{i}. {topic['Text'][:200]}")
                    if topic.get("FirstURL"):
                        clean_url = clean_duckduckgo_url(topic['FirstURL'])
                        lines.append(f" URL: {clean_url}")

        if len(lines) == 1:
            return f"[INFO] No results found for '{query}'"

        return "\n".join(lines)

    except Exception as e:
        return f"[ERROR] DuckDuckGo search failed: {e}"

def fetch_page(url, use_playwright=False):
    """
    Fetch and return the text content of a webpage

    url: the URL to fetch
    use_playwright: force Playwright even for static pages
    returns: page text content or error message
    """
    if use_playwright:
        return fetch_with_playwright(url)

    # try BeautifulSoup first — faster and lighter
    result = fetch_with_bs(url)

    # if BS returns very little content the page likely needs JS
    if result.startswith("[ERROR]") or len(result) < 200:
        if DEBUG:
            print(f"[BROWSER] BS returned little content, trying Playwright")
        return fetch_with_playwright(url)

    return result


def fetch_with_bs(url):
    """
    Fetch page content using requests and BeautifulSoup for static HTML pages

    url: the URL to fetch
    returns: cleaned page text or error message
    """
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # remove scripts, styles and nav elements — we want readable content only
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # get main content
        text = soup.get_text(separator="\n", strip=True)

        # clean up excessive blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)

        # cap at 3000 chars to avoid flooding context
        if len(cleaned) > 3000:
            cleaned = cleaned[:3000] + "\n[Content truncated]"

        return f"[PAGE: {url}]\n{cleaned}"

    except requests.exceptions.Timeout:
        return f"[ERROR] Page timed out: {url}"
    except requests.exceptions.ConnectionError:
        return f"[ERROR] Could not connect to: {url}"
    except Exception as e:
        return f"[ERROR] Could not fetch page: {e}"


def fetch_with_playwright(url):
    """
    Fetch page content using Playwright headless browser for JS heavy pages

    url: the URL to fetch
    returns: page text content or error message
    """
    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            page.wait_for_load_state("networkidle", timeout=10000)

            # extract text content
            text = page.inner_text("body")
            browser.close()

        # cap at 3000 chars
        if len(text) > 3000:
            text = text[:3000] + "\n[Content truncated]"

        return f"[PAGE: {url}]\n{text}"

    except Exception as e:
        return f"[ERROR] Playwright fetch failed: {e}"

def open_url(url):
    """
    Open a URL in the user's default browser

    url: the URL to open
    returns: success or error message
    """
    try:
        # add https if no scheme provided
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        webbrowser.open(url)
        return f"[SUCCESS] Opened {url}"
    except Exception as e:
        return f"[ERROR] Could not open URL: {e}"


def search_and_open(query):
    """
    Search the web and open the top result in the browser

    query: search query string
    returns: success message with the URL opened
    """
    results = search_web(query, num_results=1)

    # extract first URL from results
    urls = re.findall(r'URL: (https?://[^\s\]]+)', results)

    urls = [clean_duckduckgo_url(u) for u in urls]
    external_urls = [u for u in urls if "duckduckgo.com" not in u]

    if not external_urls:
        # fall back to any URL if no external ones found
        if urls:
            open_url(urls[0])
            return f"[SUCCESS] Opened: {urls[0]}"
        return f"[INFO] No results found for '{query}'"
    
    open_url(external_urls[0])
    return f"[SUCCESS] Opened top result for '{query}': {external_urls[0]}"

def clean_duckduckgo_url(url):
    """
    Strip DuckDuckGo redirect wrapper to get the actual destination URL
    DDG wraps links as: https://duckduckgo.com/l/?uddg=<encoded_url>
    
    url: potentially wrapped DDG URL
    returns: clean destination URL
    """
    if "duckduckgo.com/l/?" in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if "uddg" in params:
            return unquote(params["uddg"][0])
    return url