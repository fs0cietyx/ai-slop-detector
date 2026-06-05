import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import google.generativeai as genai
from tqdm.asyncio import tqdm

from .core.config import config, logger


@dataclass(frozen=True)
class GeneratorConfig:
    """Hardened configuration for the AI generation pipeline."""

    model_name: str = "gemini-1.5-flash"
    requests_per_minute: int = 15
    max_concurrent_tasks: int = 5
    summary_prompt: str = "Extract 5-7 key factual bullet points from this text:\n\n{text}"
    generate_prompt: str = "Write a comprehensive, encyclopedic Wikipedia-style paragraph based ONLY on these facts. Do not use conversational filler:\n\n{text}"


class AIGenerator:
    """
    Enterprise-grade AI Data Synthesis Engine.

    Adheres to Pillar I (Secrets Isolation) and Pillar V (Rate Limiting & Abuse Prevention).
    """

    def __init__(self) -> None:
        self.config = GeneratorConfig()
        
        # [AppSec] Secrets Isolation: Ensure key is extracted from SecretStr container
        if not config.GEMINI_API_KEY:
            raise RuntimeError("CRITICAL_SECURITY_FAILURE: GEMINI_API_KEY not configured.")
        
        try:
            genai.configure(api_key=config.GEMINI_API_KEY.get_secret_value())  # type: ignore[attr-defined]
            self.model = genai.GenerativeModel(self.config.model_name)  # type: ignore[attr-defined]
            logger.info(f"AIGenerator online. Architecture: {self.config.model_name}")
        except Exception as e:
            logger.critical(f"API_CONNECTION_FAILURE: Gemini handshake failed: {str(e)}")
            raise

    async def _safe_generate(self, prompt: str, semaphore: asyncio.Semaphore) -> Optional[str]:
        """
        Executes a throttled, sanitized call to the LLM engine.
        """
        # [AppSec] Payload Hardening: Enforce strict char limits before LLM egress
        sanitized_prompt = prompt.strip()[: config.MAX_INPUT_CHARS]
        
        async with semaphore:
            try:
                # [Optimization] Non-blocking thread-pool execution for synchronous SDK
                response = await asyncio.to_thread(self.model.generate_content, sanitized_prompt)
                
                if not response or not hasattr(response, "text"):
                    return None
                    
                result = str(response.text).strip()
                return result if len(result) > 50 else None

            except Exception as e:
                logger.debug(f"LLM_GENERATION_FAILURE: {str(e)}")
                return None
            finally:
                # [AppSec] Denial-of-Wallet Prevention: Enforce delay between requests
                delay = 60.0 / self.config.requests_per_minute
                await asyncio.sleep(delay)

    async def process_pipeline(self, input_file: str, output_file: str, mode: str) -> None:
        """
        Managed batch processor for large-scale text synthesis.
        """
        if not os.path.exists(input_file):
            logger.warning(f"PIPELINE_BYPASS: {input_file} missing. Skipping {mode}.")
            return

        input_file = os.path.abspath(input_file)
        output_file = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Resume logic: Deduplicate based on source URL
        processed_sources: Set[str] = set()
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        processed_sources.add(json.loads(line)["source"])
                    except Exception:
                        continue

        items: List[Dict[str, Any]] = []
        with open(input_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("source") not in processed_sources:
                        items.append(data)
                except Exception:
                    continue

        if not items:
            logger.info(f"PIPELINE_IDLE: No new records to process for {mode}.")
            return

        logger.info(f"PIPELINE_START: {mode} ({len(items)} records)")
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        prompt_template = (
            self.config.summary_prompt if "summary" in mode.lower() 
            else self.config.generate_prompt
        )

        with open(output_file, "a", encoding="utf-8") as f:
            for data in tqdm(items, desc=f"APEX_GEN_{mode.upper()}"):
                # Input source depends on pipeline stage
                text_input = data.get("text") or data.get("summary")
                if not text_input:
                    continue

                prompt = prompt_template.format(text=text_input)
                result = await self._safe_generate(prompt, semaphore)

                if result:
                    output_record = {
                        "source": data["source"],
                        "label": 1 if "generation" in mode.lower() else 0,
                    }
                    if "summary" in mode.lower():
                        output_record["summary"] = result
                    else:
                        output_record["text"] = result
                    
                    f.write(json.dumps(output_record) + "\n")


async def run_pipeline() -> None:
    """Orchestrates the two-stage synthetic data generation pipeline."""
    gen = AIGenerator()
    
    # Stage 1: Facts Extraction (Summarization)
    await gen.process_pipeline(
        input_file="data/human_data.jsonl",
        output_file="data/summaries.jsonl",
        mode="summarization"
    )
    
    # Stage 2: Slop Generation (Re-writing)
    await gen.process_pipeline(
        input_file="data/summaries.jsonl",
        output_file="data/ai_data.jsonl",
        mode="generation"
    )

if __name__ == "__main__":
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        logger.info("Shutdown signal acknowledged.")
    except Exception as e:
        logger.critical(f"FATAL_PIPELINE_CRASH: {str(e)}")
