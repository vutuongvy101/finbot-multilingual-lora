Generate exactly 70 samples for the following cell.

TASK_MODE: {task}        # one of: planning | investment | trading
LANGUAGE: {lang}          # one of: en | vi | zh
LANGUAGE_NAME: {lang_name}  # English | Vietnamese | Simplified Chinese

Constraints for this batch:
- All 70 samples MUST be for task_mode = {task} and written entirely in {lang_name}.
- Distribute profile enum values roughly evenly within the batch.
- At least 10 samples must include unknown_fields (1-2 missing fields each).
- Every sample must satisfy the QUANTITATIVE REQUIREMENTS from the system prompt.
- Vary the chosen instruments / tickers / formulas across samples; do not repeat the same allocation pattern more than twice.

Return ONLY the JSON array of 70 objects. No surrounding text.