"""
Telegram bot for GenAI assessment: RAG (/ask) + Image captioning (/image).
Uses existing query_unstructured (RAG) and image_caption (vision) modules.
Features: message history (last 3 per user), query cache, source snippets, /summarize.
"""
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

from telegram import Update
from telegram.error import Conflict
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Project imports (run from chatbot-dev directory)
from query_unst import query_unstructured
from image_caption import caption_image
from config.config_validator import validate_ini_file
from utils import query_openai_mult

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CONFIG_LOC = os.environ.get("CONFIG_LOC", str(Path(__file__).parent / "config" / "config.ini"))
CONFIG = validate_ini_file(CONFIG_LOC)

# Bot token: prefer env so it's not committed
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or CONFIG.get("telegram", {}).get("bot_token", "")

# Last N interactions per user (message history awareness)
HISTORY_SIZE = 3
# In-memory cache: normalized query -> { "text", "citations" } (don't re-embed seen queries)
_query_cache = {}
# Max cache size to avoid unbounded growth
_MAX_CACHE_SIZE = 500

def _normalize_query(q: str) -> str:
    return q.strip().lower() if q else ""

def _format_source_snippets(citations: list, max_snippet_len: int = 180) -> str:
    """Format RAG citations as 'Source: filename — snippet' lines."""
    if not citations:
        return ""
    lines = []
    for c in citations:
        fn = c.get("filename") or c.get("label") or "document"
        desc = (c.get("description") or "").strip()
        if len(desc) > max_snippet_len:
            desc = desc[: max_snippet_len].rsplit(" ", 1)[0] + "…"
        lines.append(f"• {fn} — \"{desc}\"")
    return "📎 Source snippets:\n" + "\n".join(lines)

def _get_recent_chat_str(history: list) -> str:
    """Build a short string of last Q&A for RAG context."""
    if not history:
        return ""
    parts = []
    for h in history[-3:]:
        u = h.get("user") or h.get("query") or ""
        a = (h.get("bot") or h.get("answer") or "")[: 200]
        if u or a:
            parts.append(f"Q: {u}\nA: {a}")
    return "\n\n".join(parts) if parts else ""

def _append_history(context: ContextTypes.DEFAULT_TYPE, entry_type: str, user_content: str, bot_content: str) -> None:
    """Keep last HISTORY_SIZE interactions per user."""
    hist = context.user_data.get("history") or []
    hist.append({"type": entry_type, "user": user_content, "bot": bot_content})
    context.user_data["history"] = hist[-HISTORY_SIZE:]

HELP_TEXT = """
🤖 *GenAI Bot* — RAG + Image captioning

*Commands:*
/ask _&lt;query&gt;_ — Ask a question (RAG over knowledge base)
  Example: `/ask What is the leave policy?`

/image — Send or reply with a photo to get a caption and tags
  Upload an image and I'll describe it and suggest 3 tags.

/summarize — Summarize your last image caption or last chat (up to last 3 Q&amp;As)

/help — Show this message

*Tips:*
• I keep the last 3 interactions per user for context.
• Repeated questions use cache (no re-embedding).
• /ask replies include source snippets (which doc was used).
"""


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Usage: /ask &lt;your question&gt;\nExample: /ask What is the leave policy?",
            parse_mode="HTML",
        )
        return
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Please provide a question after /ask.")
        return

    await update.message.reply_chat_action("typing")
    user_id = str(update.effective_user.id) if update.effective_user else "telegram_user"
    email = f"telegram.{user_id}@bot.user"
    correlation_id = f"tg-{update.update_id}"
    outside_context = "no"
    verbosity = "medium"

    # Build recent chat for RAG context (message history awareness)
    hist = context.user_data.get("history") or []
    recent_chat = _get_recent_chat_str(hist)

    # Basic caching: don't re-embed already seen queries
    norm = _normalize_query(query)
    if norm and norm in _query_cache:
        cached = _query_cache[norm]
        text = cached.get("text") or "No answer generated."
        citations = cached.get("citations") or []
        snippets = _format_source_snippets(citations)
        if snippets:
            text = f"{text}\n\n{snippets}"
        if len(text) > 4000:
            text = text[:3990] + "\n…"
        await update.message.reply_text(text)
        _append_history(context, "ask", query, text)
        return

    try:
        result_str = await asyncio.to_thread(
            query_unstructured,
            query,
            outside_context,
            email,
            correlation_id,
            verbosity,
            recent_chat_history=recent_chat if recent_chat else None,
        )
        result = json.loads(result_str)
        text = result.get("textdata") or result.get("text", "No answer generated.")
        citations = result.get("citation") or []
        if isinstance(citations, str):
            citations = []
        # Source snippets: show which doc was used in RAG response
        snippets = _format_source_snippets(citations)
        if snippets:
            text = f"{text}\n\n{snippets}"
        # Update cache (don't re-embed same query again)
        if norm and isinstance(citations, list):
            if len(_query_cache) >= _MAX_CACHE_SIZE:
                # Drop oldest (first) entry; dict is insertion-ordered in 3.7+
                _query_cache.pop(next(iter(_query_cache)), None)
            _query_cache[norm] = {"text": result.get("textdata") or result.get("text", ""), "citations": citations}
        if len(text) > 4000:
            text = text[:3990] + "\n…"
        await update.message.reply_text(text)
        _append_history(context, "ask", query, text)
    except Exception as e:
        logger.exception("RAG query failed")
        await update.message.reply_text(
            f"Sorry, something went wrong while answering.\nError: {str(e)[:200]}"
        )


async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Photo can be in the message or in the replied-to message
    photo = None
    if update.message.photo:
        photo = update.message.photo[-1]
    elif update.message.reply_to_message and update.message.reply_to_message.photo:
        photo = update.message.reply_to_message.photo[-1]
    if not photo:
        await update.message.reply_text(
            "Please send an image (photo) or reply to an image with /image."
        )
        return

    await update.message.reply_chat_action("typing")
    try:
        file = await context.bot.get_file(photo.file_id)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            path = tmp.name
        try:
            caption_text, tags = await asyncio.to_thread(caption_image, path)
            tags_str = ", ".join(f"#{t}" for t in tags)
            reply = f"🖼 Caption: {caption_text}\n\n🏷 Tags: {tags_str}"
            if len(reply) > 4000:
                reply = reply[:3990] + "\n…"
            await update.message.reply_text(reply)
            _append_history(context, "image", "image", reply)
        finally:
            if os.path.isfile(path):
                os.unlink(path)
    except Exception as e:
        logger.exception("Image caption failed")
        await update.message.reply_text(
            f"Sorry, I couldn't describe the image.\nError: {str(e)[:200]}"
        )


async def cmd_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Summarize last image caption or last chat (up to last 3 Q&As)."""
    hist = context.user_data.get("history") or []
    if not hist:
        await update.message.reply_text(
            "No recent activity to summarize. Use /ask or send an /image first."
        )
        return
    last = hist[-1]
    if last.get("type") == "image":
        content = last.get("bot") or ""
        prompt = f"Summarize the following image caption and tags in one short sentence (under 25 words):\n\n{content}"
    else:
        parts = []
        for h in hist[-HISTORY_SIZE:]:
            u, a = h.get("user") or "", (h.get("bot") or "")[: 300]
            if u or a:
                parts.append(f"Q: {u}\nA: {a}")
        content = "\n\n".join(parts)
        prompt = f"Summarize the following conversation in 2-3 short sentences:\n\n{content}"
    await update.message.reply_chat_action("typing")
    try:
        summary = await asyncio.to_thread(query_openai_mult, prompt, 0)
        if not summary:
            summary = "Could not generate summary."
        if len(summary) > 1000:
            summary = summary[:990] + "\n…"
        await update.message.reply_text(f"📋 Summary:\n{summary}")
    except Exception as e:
        logger.exception("Summarize failed")
        await update.message.reply_text(
            f"Sorry, I couldn't summarize that.\nError: {str(e)[:150]}"
        )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I'm a GenAI bot with RAG and image captioning.\n\n" + HELP_TEXT,
        parse_mode="Markdown",
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Conflict: only one bot instance can poll at a time."""
    if isinstance(context.error, Conflict):
        logger.error(
            "Conflict: Another bot instance is already running (same token). "
            "Stop all other telegram_bot.py processes (other terminals, IDE run, etc.) "
            "and run only one instance."
        )
        await context.application.stop()
    else:
        logger.exception("Update %s caused error: %s", update, context.error)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "Set TELEGRAM_BOT_TOKEN in environment or add [telegram] bot_token in config."
        )
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("summarize", cmd_summarize))
    logger.info("Bot started. Polling. (Run only one instance per bot token.)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
