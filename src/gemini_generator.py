import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

import google.generativeai as genai
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Pillar VI: Secure Observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Pillar I: Secrets Isolation
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_key_here":
    logger.critical("ACCESS DENIED: GEMINI_API_KEY missing.")
    sys.exit(1)

# Configuration
HUMAN_DATA_FILE: str = os.path.abspath("data/human_data.jsonl")
SUMMARIES_FILE: str = os.path.abspath("data/summaries.jsonl")
AI_DATA_FILE: str = os.path.abspath("data/ai_data.jsonl")
MODEL_NAME: str = "gemini-1.5-flash"

# Pillar V: Rate Limiting
REQUESTS_PER_MINUTE: int = 15
DELAY_BETWEEN_REQUESTS: float = 60 / REQUESTS_PER_MINUTE

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gen_model = genai.GenerativeModel(MODEL_NAME)
except Exception:
    logger.critical("Failed to connect to Gemini API.")
    sys.exit(1)

async def call_gemini(prompt: str, semaphore: asyncio.Semaphore) -> Optional[str]:
    """
    Calls Gemini API with strict sanitization and concurrency guards.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return None
        
    # Pillar IV: Sanitization
    prompt = "".join(char for char in prompt if ord(char) >= 32 or char in "\n\r\t")
    
    async with semaphore:
        try:
            response: Any = await asyncio.to_thread(gen_model.generate_content, prompt)
            if not response or not hasattr(response, 'text'):
                return None
            return str(response.text.strip())
        except Exception:
            return None
        finally:
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

async def process_batch(input_path: str, output_path: str, prompt_template: str, desc: str) -> None:
    """
    Managed batch processor for high-volume AI generation.
    """
    if not os.path.isfile(input_path):
        return

    processed_sources: Set[str] = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_sources.add(json.loads(line)['source'])
                except (json.JSONDecodeError, KeyError):
                    continue

    items_to_process: List[Dict[str, Any]] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data: Dict[str, Any] = json.loads(line)
                if 'source' in data and data['source'] not in processed_sources:
                    items_to_process.append(data)
            except json.JSONDecodeError:
                continue

    if not items_to_process:
        return

    logger.info(f"Processing {len(items_to_process)} records for {desc}...")
    semaphore = asyncio.Semaphore(5)
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for data in tqdm(items_to_process, desc=desc):
            input_text: str = str(data.get('text') or data.get('summary', ''))
            prompt: str = prompt_template.format(text=input_text)
            
            result = await call_gemini(prompt, semaphore)
            
            if result:
                output_record: Dict[str, Any] = {
                    "source": data['source'],
                    "label": 1 if "generation" in desc.lower() else 0
                }
                if "summarize" in desc.lower():
                    output_record["summary"] = result
                else:
                    output_record["text"] = result
                    
                f.write(json.dumps(output_record) + '\n')

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    s_prompt: str = "Extract factual points from this text:\n\n{text}"
    g_prompt: str = "Write a Wikipedia-style paragraph from these facts:\n\n{text}"
    
    try:
        loop.run_until_complete(process_batch(HUMAN_DATA_FILE, SUMMARIES_FILE, s_prompt, "Summarizing"))
        loop.run_until_complete(process_batch(SUMMARIES_FILE, AI_DATA_FILE, g_prompt, "Generating"))
    except KeyboardInterrupt:
        logger.info("Shutdown initiated.")
    except Exception:
        logger.critical("Fatal pipeline failure.")
