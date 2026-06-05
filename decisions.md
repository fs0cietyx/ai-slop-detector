# Key Decisions

## 1. Wikipedia as Primary Human Source
- **Decision**: Use Wikipedia for the human-written text dataset.
- **Rationale**: Easy-to-use API, linked structure simplifies crawling, and provides a "stable" genre for the proof of concept. 
- **Revisit**: When moving beyond POC to a multi-domain detector.

## 2. Two-Pass AI Generation (Summary -> Paragraph)
- **Decision**: Instead of direct rewriting, first summarize key points, then write a paragraph from those points.
- **Rationale**: Prevents the AI from simply mimicking human sentence structure/phrasing from the source. It better models how users actually use AI to generate content from facts.
- **Revisit**: If the "slop" patterns change with newer model versions.

## 3. Cost Optimization: GPT-5 Nano & Batch API
- **Decision**: Use the smallest capable model (GPT-5 Nano) and OpenAI's Batch API.
- **Rationale**: Reduces costs by ~50% and handles large-scale generation asynchronously.
- **Revisit**: If model quality becomes a bottleneck for detection accuracy.

## 6. Pivot to RAID Dataset for Training
- **Decision**: Use the `liamdugan/raid` dataset as the primary training source instead of the custom Gemini-generated data.
- **Rationale**: RAID is a high-quality, large-scale benchmark with multi-domain and multi-model coverage. It also includes adversarial attacks, making the resulting detector significantly more robust.
- **Validation**: Our original 10,000 Wikipedia paragraphs will be repurposed as a "domain-specific" validation set to test out-of-distribution generalization.
