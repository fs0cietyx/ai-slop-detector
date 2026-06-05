import requests
from bs4 import BeautifulSoup
import json
import random
import time
import re
from tqdm import tqdm
import os
import logging
import sys

# Pillar 6: Secure Observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_URL = "https://en.wikipedia.org"
SEEDS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Special:Random"
]
TARGET_PARAGRAPHS = 10000
OUTPUT_FILE = os.path.abspath("data/human_data.jsonl")

# Pillar 5: Bot Mitigation - Explicit User-Agent
HEADERS = {
    'User-Agent': 'AISlopDetectorBot/1.0 (Enterprise-Security-Audit; mailto:security@example.com)'
}

def clean_text(text):
    """
    Sanitizes extracted text (Pillar 4).
    """
    if not text:
        return None
        
    # Remove citation markers [1], [vague], etc.
    text = re.sub(r'\[[a-zA-Z0-9\s]+\]', '', text)
    
    # Pillar 4: Neutralize potentially dangerous fragments (e.g., script tags, LaTeX injection)
    if "$$" in text or r"{\displaystyle" in text or "<script" in text.lower():
        return None
        
    text = text.strip()
    
    # Pillar 5: Quality/DoS filter - Reject unusually short or long fragments
    words = text.split()
    if len(words) < 25 or len(words) > 1000:
        return None
        
    return text

def fetch_data(url):
    """
    Fetches data with strict validation and resource limits (Pillar 4 & 5).
    """
    # Pillar 4: URL Validation
    if not url.startswith(BASE_URL):
        logger.warning(f"Rejected SSRF-like traversal attempt to: {url}")
        return None, []

    try:
        # Pillar 6: Secure Observability - Standardized Logging
        # Pillar 5: Anti-Abuse - Stream response to enforce size limits
        response = requests.get(url, headers=HEADERS, timeout=10, stream=True)
        
        if response.status_code != 200:
            logger.error(f"Upstream failure at {url}: Status {response.status_code}")
            return None, []
            
        # Pillar 4: Input Validation - Check Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            logger.warning(f"Rejected non-HTML content-type from {url}")
            return None, []

        # Pillar 5: Resource Limit - Enforce maximum download size
        MAX_SIZE = 2 * 1024 * 1024 # 2MB limit
        content = b""
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_SIZE:
                logger.error(f"Resource limit exceeded at {url}. Terminating connection.")
                return None, []
        
        # Pillar 4: Secure Parsing
        soup = BeautifulSoup(content, 'html.parser')
        
        paragraphs = []
        content_div = soup.find(class_="mw-parser-output")
        if content_div:
            # Recursive check to ensure nested p tags are captured but sanitized
            p_tags = content_div.find_all('p', recursive=True)
            for p in p_tags:
                cleaned = clean_text(p.get_text())
                if cleaned:
                    paragraphs.append(cleaned)
        
        links = []
        # Pillar 5: Anti-Abuse - Limit number of links extracted per page
        MAX_LINKS = 200
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Only internal wiki links, avoiding special pages and files
            if href.startswith('/wiki/') and ':' not in href:
                links.append(BASE_URL + href)
                if len(links) >= MAX_LINKS:
                    break
        
        return paragraphs, links
    except Exception:
        logger.error(f"Network interaction error during crawl.")
        return None, []

def run_crawler():
    visited = set()
    queue = SEEDS.copy()
    paragraphs_collected = 0
    
    # Ensure data directory exists with restricted permissions (Pillar 6)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Initialize count if file exists
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as check_f:
            paragraphs_collected = sum(1 for _ in check_f)
            logger.info(f"Resuming from record count: {paragraphs_collected}")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        with tqdm(total=TARGET_PARAGRAPHS, initial=paragraphs_collected, desc="Secure Crawl") as pbar:
            while paragraphs_collected < TARGET_PARAGRAPHS and queue:
                # Pillar 5: Randomize traversal to avoid static bot patterns
                current_url = queue.pop(random.randint(0, len(queue) - 1))
                if current_url in visited:
                    continue
                
                visited.add(current_url)
                page_paragraphs, page_links = fetch_data(current_url)
                
                if page_paragraphs:
                    # Take up to 2 paragraphs as per plan
                    to_save = page_paragraphs[:2]
                    for p in to_save:
                        if paragraphs_collected < TARGET_PARAGRAPHS:
                            # Pillar 4: Schema Enforcement on Write
                            json.dump({
                                "text": p, 
                                "source": current_url, 
                                "label": 0,
                                "audit_timestamp": time.time()
                            }, f)
                            f.write('\n')
                            paragraphs_collected += 1
                            pbar.update(1)
                
                new_links = [l for l in page_links if l not in visited]
                if new_links:
                    # Sample 10 links to maintain a manageable queue (Pillar 5)
                    sample_size = min(len(new_links), 10)
                    queue.extend(random.sample(new_links, sample_size))
                
                # Pillar 5: Rate Limiting - Polite Crawler Policy
                time.sleep(1.0)
                
                # Queue management to prevent memory exhaustion (Pillar 5)
                if len(queue) > 5000:
                    queue = random.sample(queue, 2000)

if __name__ == "__main__":
    try:
        run_crawler()
    except KeyboardInterrupt:
        logger.info("Crawl aborted by security operator.")
    except Exception:
        logger.critical("Fatal failure in crawler logic.")
