import json
import logging
import os
import random
import re
import sys
import time
from typing import List, Tuple, Set, Optional, Dict, Any

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

# Pillar VI: Secure Observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_URL: str = "https://en.wikipedia.org"
SEEDS: List[str] = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Special:Random"
]
TARGET_PARAGRAPHS: int = 10000
OUTPUT_FILE: str = os.path.abspath("data/human_data.jsonl")

# Pillar V: Bot Mitigation
HEADERS: Dict[str, str] = {
    'User-Agent': 'AISlopDetectorBot/1.0 (Enterprise-Quality-Audit; mailto:security@example.com)'
}

def clean_text(text: Optional[str]) -> Optional[str]:
    """
    Sanitizes extracted text with strict quality boundaries.
    """
    if not text:
        return None
        
    # Remove citations
    text = re.sub(r'\[[a-zA-Z0-9\s]+\]', '', text)
    
    # Pillar IV: Payload Neutralization
    if "$$" in text or r"{\displaystyle" in text or "<script" in text.lower():
        return None
        
    text = text.strip()
    
    # Quality filter
    words = text.split()
    if len(words) < 25 or len(words) > 1000:
        return None
        
    return text

def fetch_data(url: str) -> Tuple[List[str], List[str]]:
    """
    Fetches and validates external data with resource limits.
    """
    # Pillar IV: SSRF Protection
    if not url.startswith(BASE_URL):
        logger.warning(f"Blocked unauthorized traversal to: {url}")
        return [], []

    try:
        response = requests.get(url, headers=HEADERS, timeout=10, stream=True)
        
        if response.status_code != 200:
            return [], []
            
        # Pillar IV: Content-Type Enforcement
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            return [], []

        # Pillar V: Memory Guard
        MAX_SIZE = 2 * 1024 * 1024 # 2MB
        content = b""
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_SIZE:
                return [], []
        
        soup = BeautifulSoup(content, 'html.parser')
        
        paragraphs: List[str] = []
        content_div = soup.find(class_="mw-parser-output")
        if isinstance(content_div, Tag):
            p_tags = content_div.find_all('p', recursive=True)
            for p in p_tags:
                cleaned = clean_text(p.get_text())
                if cleaned:
                    paragraphs.append(cleaned)
        
        links: List[str] = []
        MAX_LINKS = 200
        for link in soup.find_all('a', href=True):
            href: str = str(link['href'])
            if href.startswith('/wiki/') and ':' not in href:
                links.append(BASE_URL + href)
                if len(links) >= MAX_LINKS:
                    break
        
        return paragraphs, links
    except Exception:
        return [], []

def run_crawler() -> None:
    """
    Main asynchronous traversal logic for data collection.
    """
    visited: Set[str] = set()
    queue: List[str] = SEEDS.copy()
    paragraphs_collected: int = 0
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as check_f:
            paragraphs_collected = sum(1 for _ in check_f)
            logger.info(f"Crawl resume sequence initiated at count: {paragraphs_collected}")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        with tqdm(total=TARGET_PARAGRAPHS, initial=paragraphs_collected, desc="Secure Crawl") as pbar:
            while paragraphs_collected < TARGET_PARAGRAPHS and queue:
                current_url = queue.pop(random.randint(0, len(queue) - 1)) # nosec
                if current_url in visited:
                    continue
                
                visited.add(current_url)
                page_paragraphs, page_links = fetch_data(current_url)
                
                if page_paragraphs:
                    to_save = page_paragraphs[:2]
                    for p in to_save:
                        if paragraphs_collected < TARGET_PARAGRAPHS:
                            json.dump({
                                "text": p, 
                                "source": current_url, 
                                "label": 0,
                                "timestamp": time.time()
                            }, f)
                            f.write('\n')
                            paragraphs_collected += 1
                            pbar.update(1)
                
                new_links = [link_url for link_url in page_links if link_url not in visited]
                if new_links:
                    sample_size = min(len(new_links), 10)
                    queue.extend(random.sample(new_links, sample_size)) # nosec
                
                time.sleep(1.0)
                
                if len(queue) > 5000:
                    queue = random.sample(queue, 2000) # nosec

if __name__ == "__main__":
    try:
        run_crawler()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by operator.")
    except Exception:
        logger.critical("Fatal execution failure in crawler.")
