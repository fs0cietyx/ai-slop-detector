import os
import json
import asyncio
import re
from tqdm.asyncio import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
import time
import logging
import sys

# Pillar 6: Secure Observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Pillar 1: Secrets Management - Strict Variable Parsing
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_key_here":
    logger.critical("ACCESS DENIED: GEMINI_API_KEY not found in environment.")
    sys.exit(1)

# Configuration
HUMAN_DATA_FILE = os.path.abspath("data/human_data.jsonl")
SUMMARIES_FILE = os.path.abspath("data/summaries.jsonl")
AI_DATA_FILE = os.path.abspath("data/ai_data.jsonl")
MODEL_NAME = "gemini-1.5-flash"

# Pillar 5: Strict Rate Limiting (Hardcoded per Tier)
REQUESTS_PER_MINUTE = 15
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

# Secure API Configuration
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
except Exception:
    logger.critical("Failed to initialize cryptographic connection to Gemini API.")
    sys.exit(1)

async def call_gemini(prompt, semaphore):
    """
    Calls Gemini API with strict sanitization (Pillar 4) and rate limiting (Pillar 5).
    """
    # Pillar 4: Aggressive Input Sanitization
    if not isinstance(prompt, str) or not prompt.strip():
        logger.warning("Rejected malformed AI prompt payload.")
        return None
        
    # Neutralize control characters
    prompt = "".join(char for char in prompt if ord(char) >= 32 or char in "\n\r\t")
    
    async with semaphore:
        try:
            # Pillar 5: Concurrency Guard
            response = await asyncio.to_thread(model.generate_content, prompt)
            
            # Validate Response Integrity
            if not response or not hasattr(response, 'text') or not response.text:
                logger.error("Incomplete response received from API.")
                return None
                
            return response.text.strip()
        except Exception:
            # Pillar 6: Secure Logging - Do not leak API error details which might contain secrets
            logger.error("External API interaction failed.")
            return None
        finally:
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

async def process_batch(input_path, output_path, prompt_template, desc):
    """
    Generalized batch processor with strict schema validation (Pillar 4).
    """
    if not os.path.isfile(input_path):
        logger.error(f"Input vector missing: {input_path}")
        return

    # Load existing to prevent duplicate effort/API cost
    processed_sources = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_sources.add(json.loads(line)['source'])
                except (json.JSONDecodeError, KeyError):
                    continue

    items_to_process = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Pillar 4: Schema Validation
                if 'source' in data and ('text' in data or 'summary' in data):
                    if data['source'] not in processed_sources:
                        items_to_process.append(data)
            except json.JSONDecodeError:
                logger.warning("Dropped malformed JSON record during ingestion.")
                continue

    if not items_to_process:
        logger.info(f"Batch {desc}: No new payloads identified.")
        return

    logger.info(f"Initiating {desc} for {len(items_to_process)} records...")
    semaphore = asyncio.Semaphore(5) # Pillar 5: Concurrency Limit
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for data in tqdm(items_to_process, desc=desc):
            input_text = data.get('text') or data.get('summary')
            prompt = prompt_template.format(text=input_text)
            
            result = await call_gemini(prompt, semaphore)
            
            if result:
                # Construct clean output schema
                output_record = {
                    "source": data['source'],
                    "label": 1 if "generation" in desc.lower() else 0
                }
                if "summarize" in desc.lower():
                    output_record["summary"] = result
                    output_record["original_text"] = input_text
                else:
                    output_record["text"] = result
                    
                f.write(json.dumps(output_record) + '\n')

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    
    summarize_prompt = "Extract key factual points from this text as concise bullet points:\n\n{text}"
    generate_prompt = "Write a professional Wikipedia-style paragraph based ONLY on these facts:\n\n{text}"
    
    try:
        logger.info("--- Phase 1: Secure Summarization ---")
        loop.run_until_complete(process_batch(HUMAN_DATA_FILE, SUMMARIES_FILE, summarize_prompt, "Summarizing"))
        
        logger.info("\n--- Phase 2: Secure Generation ---")
        loop.run_until_complete(process_batch(SUMMARIES_FILE, AI_DATA_FILE, generate_prompt, "Generating"))
    except KeyboardInterrupt:
        logger.info("System shutdown initiated by administrator.")
    except Exception as e:
        logger.critical(f"Panic: System-wide failure in generation pipeline.")
