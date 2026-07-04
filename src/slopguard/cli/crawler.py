import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import httpx
from bs4 import BeautifulSoup, Tag
from tqdm.asyncio import tqdm

from slopguard.core.config import logger


@dataclass(frozen=True)
class CrawlerConfig:
    """Immutable configuration for the asynchronous crawler."""

    base_url: str = "https://en.wikipedia.org"
    seeds: Tuple[str, ...] = (
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Special:Random",
    )
    target_count: int = 10000
    max_payload_size: int = 1 * 1024 * 1024  # 1MB per page
    request_timeout: float = 10.0
    user_agent: str = "AISlopDetectorBot/1.0 (+https://github.com/fs0cietyx/ai-slop-detector)"


class AsyncCrawler:
    """
    Enterprise-grade Asynchronous Web Crawler.

    Implements non-blocking I/O, SSRF protection, and memory-safe buffering.
    Adheres to Pillar III (Asynchronous Mastery) and Pillar IV (Defensive Programming).
    """

    def __init__(self, output_file: str) -> None:
        self.config = CrawlerConfig()
        self.output_file = os.path.abspath(output_file)
        self.visited: Set[str] = set()
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.collected_count: int = 0
        self._lock = asyncio.Lock()

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def _sanitize_text(self, text: str) -> Optional[str]:
        """
        Refined text sanitization and quality filtering.
        
        Eliminates citations, LaTeX noise, and low-entropy sequences.
        """
        # Remove Wikipedia citations like [1], [23], [citation needed]
        text = re.sub(r"\[[0-9a-zA-Z\s]+\]", "", text)
        
        # Neutralize Potential Exploit Patterns (LaTeX injections, script tags)
        if any(p in text.lower() for p in ["{\displaystyle", "<script", "eval("]):
            return None

        clean = text.strip()
        words = clean.split()
        
        # Entropy filter: Ensure text is substantial enough for ML training
        if len(words) < 25 or len(words) > 800:
            return None
            
        return clean

    async def _fetch_page(self, client: httpx.AsyncClient, url: str) -> Tuple[List[str], List[str]]:
        """
        Securely fetches a page with SSRF protection and memory guards.
        """
        # [AppSec] SSRF Protection: Enforce strict domain boundaries
        if not url.startswith(self.config.base_url):
            logger.warning(f"SSRF_PREVENTION: Blocked unauthorized traversal to {url}")
            return [], []

        try:
            async with client.stream("GET", url, follow_redirects=True) as response:
                if response.status_code != 200:
                    return [], []

                # [AppSec] Content-Type Enforcement
                if "text/html" not in response.headers.get("content-type", "").lower():
                    return [], []

                # [AppSec] Resource Discipline: Read with strict byte-size limit
                content = b""
                async for chunk in response.aiter_bytes():
                    content += chunk
                    if len(content) > self.config.max_payload_size:
                        logger.debug(f"RESOURCE_LIMIT: Dropping large page {url}")
                        return [], []

                soup = BeautifulSoup(content, "html.parser")
                
                # Extract clean paragraphs
                paragraphs: List[str] = []
                main_content = soup.find(class_="mw-parser-output")
                if isinstance(main_content, Tag):
                    for p in main_content.find_all("p", recursive=True):
                        clean = self._sanitize_text(p.get_text())
                        if clean:
                            paragraphs.append(clean)

                # Extract valid wiki links
                links: List[str] = []
                for link in soup.find_all("a", href=True):
                    href = str(link["href"])
                    if href.startswith("/wiki/") and ":" not in href:
                        links.append(self.config.base_url + href)
                        if len(links) >= 100:  # Bound BFS branching factor
                            break
                
                return paragraphs, links

        except Exception as e:
            logger.debug(f"FETCH_ERROR: {url} failed: {str(e)}")
            return [], []

    async def _worker(self, worker_id: int, pbar: tqdm) -> None:
        """Concurrent worker processing the crawl queue."""
        headers = {"User-Agent": self.config.user_agent}
        
        async with httpx.AsyncClient(headers=headers, timeout=self.config.request_timeout) as client:
            while self.collected_count < self.config.target_count:
                try:
                    url = await asyncio.wait_for(self.queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    break

                if url in self.visited:
                    self.queue.task_done()
                    continue

                self.visited.add(url)
                paragraphs, links = await self._fetch_page(client, url)

                # Save a subset of paragraphs to prevent over-representing a single page
                if paragraphs:
                    async with self._lock:
                        remaining = self.config.target_count - self.collected_count
                        to_save = paragraphs[: min(len(paragraphs), 3, remaining)]
                        
                        with open(self.output_file, "a", encoding="utf-8") as f:
                            for p in to_save:
                                record = {
                                    "text": p,
                                    "source": url,
                                    "label": 0,
                                    "metadata": {"crawler": "AsyncCrawler/1.0"}
                                }
                                f.write(json.dumps(record) + "\n")
                                self.collected_count += 1
                                pbar.update(1)

                # BFS: Extend queue with discovered links
                new_links = [link_url for link_url in links if link_url not in self.visited]
                if new_links:
                    for link_url in random.sample(new_links, min(len(new_links), 5)):
                        await self.queue.put(link_url)

                self.queue.task_done()
                
                # Ethical Scraping: Rate limiting
                await asyncio.sleep(0.5)

    async def run(self, num_workers: int = 10) -> None:
        """Orchestrates the asynchronous crawl lifecycle."""
        logger.info(f"Initiating Secure Crawl: Target={self.config.target_count}")
        
        # Resume sequence
        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                self.collected_count = sum(1 for _ in f)
            logger.info(f"Resume sequence active. Progress: {self.collected_count}")

        for seed in self.config.seeds:
            await self.queue.put(seed)

        with tqdm(total=self.config.target_count, initial=self.collected_count, desc="DATA_CRAWL") as pbar:
            workers = [asyncio.create_task(self._worker(i, pbar)) for i in range(num_workers)]
            await asyncio.gather(*workers)

        logger.info("Crawl lifecycle finalized successfully.")


if __name__ == "__main__":
    crawler = AsyncCrawler(output_file="data/human_data.jsonl")
    try:
        asyncio.run(crawler.run(num_workers=15))
    except KeyboardInterrupt:
        logger.info("Shutdown signal acknowledged.")
    except Exception as e:
        logger.critical(f"FATAL_CRAWLER_CRASH: {str(e)}")
