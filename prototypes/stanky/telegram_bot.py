import logging
import os
from collections import defaultdict
from typing import Any

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from brain_client import BrainClientError, SYSTEM_PROMPT, generate_response

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 16

chat_histories: dict[int, list[dict[str, Any]]] = defaultdict(list)


def _new_history() -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def _get_or_create_history(chat_id: int) -> list[dict[str, Any]]:
    if not chat_histories[chat_id]:
        chat_histories[chat_id] = _new_history()
    return chat_histories[chat_id]


def _truncate_history(history: list[dict[str, Any]]) -> None:
    # Keep the system message and only the most recent chat turns.
    max_total = 1 + MAX_HISTORY_MESSAGES
    if len(history) > max_total:
        history[:] = [history[0], *history[-MAX_HISTORY_MESSAGES:]]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = _new_history()
    await update.message.reply_text("Study Buddy is ready. Send a message anytime.")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = _new_history()
    await update.message.reply_text("Chat history reset.")


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return

    user_text = update.message.text
    if not user_text:
        return

    chat_id = update.effective_chat.id
    history = _get_or_create_history(chat_id)
    history.append({"role": "user", "content": user_text})
    _truncate_history(history)

    import asyncio
    try:
        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(
            None, generate_response, list(history)
        )
        history.append({"role": "assistant", "content": reply})
        _truncate_history(history)
        await update.message.reply_text(reply)
    except BrainClientError:
        logger.exception("Ollama request failed")
        await update.message.reply_text(
            "Model server is currently unavailable. Please try again later."
        )


async def ignore_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # V1 scope: ignore non-text content with no reply.
    return


def main() -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    model_name = os.getenv("OLLAMA_MODEL", "stanky3")
    server_url = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
    logger.info("Using Ollama model: %s", model_name)
    logger.info("Using Ollama endpoint: %s", server_url)

    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(~filters.TEXT, ignore_non_text))

    logger.info("Telegram bot polling started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
