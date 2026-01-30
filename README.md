<<<<<<< HEAD
# chatbot-dev

Development Phase — Python 3.10

## Quick start (Flask web app)

1. Create a new Python environment.
2. Install dependencies: `pip install -r requirements_2.txt`
3. Run: `python main.py` (or `python deploy_fix.py` as previously)

---

## Telegram GenAI Bot (RAG + Image captioning)

A lightweight GenAI bot that supports:

- **/ask &lt;query&gt;** — RAG: answers from your knowledge base (same backend as the web app)
- **/image** — Vision: send a photo to get a short caption and 3 tags (BLIP model)
- **/help** — Usage instructions

### How to run the Telegram bot locally

1. **Install dependencies** (if not already):
   ```bash
   pip install -r requirements_2.txt
   ```

2. **Bot token**  
   The bot token is read from:
   - Environment variable: `TELEGRAM_BOT_TOKEN` (recommended in production)
   - Or `config/config.ini` → `[telegram]` → `bot_token`

3. **Run the bot** (from the `chatbot-dev` project root):
   ```bash
   python telegram_bot.py
   ```
   The bot uses long polling. Talk to your bot in Telegram with `/start`, `/help`, `/ask ...`, and `/image` (with a photo).

### Models and APIs used

| Component | Model / API |
|-----------|-------------|
| **RAG (/ask)** | Existing pipeline: Azure OpenAI (config in `config/config.ini`), FAISS vector store, your uploaded documents |
| **Image caption (/image)** | Hugging Face `Salesforce/blip-image-captioning-base` (local; first run downloads the model) |

### Optional: run without committing the token

```bash
set TELEGRAM_BOT_TOKEN=your_bot_token_here
python telegram_bot.py
```

On Linux/macOS: `export TELEGRAM_BOT_TOKEN=your_bot_token_here`

### System design (high level)

```
User (Telegram)
    → python-telegram-bot (telegram_bot.py)
        → /ask  → query_unst.query_unstructured → RAG (FAISS + Azure OpenAI)
        → /image → image_caption.caption_image → BLIP (Hugging Face)
    ← Reply (text / caption + tags)
```

---

## Project structure (relevant files)

- `main.py` — Flask web app (upload, chat, health)
- `telegram_bot.py` — Telegram bot entrypoint
- `query_unst.py` — RAG query logic
- `doc_intel_search.py` — Embeddings, vector store, document search
- `image_caption.py` — Image captioning (BLIP + tag extraction)
- `config/config.ini` — Config (including optional `[telegram]`)
=======
# Abibo-AI-Task-
A telegram chatbot , well explained text to tect and imagee to text working using RAG
>>>>>>> 6269180f8bfd9d95e96e304ac56a573cc838c8ba
