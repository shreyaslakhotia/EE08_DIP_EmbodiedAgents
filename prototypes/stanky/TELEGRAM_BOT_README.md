Telegram Bot Demo Interface

Overview
- This folder now includes a reusable Ollama client in brain_client.py and a Telegram fallback interface in telegram_bot.py.
- The existing Pi app in stanky.py still runs and now uses the shared brain client.

Files
- brain_client.py: Stateless Ollama HTTP client.
- telegram_bot.py: Polling Telegram bot with per-chat in-memory history.
- stanky.py: Existing Pi GUI app, updated to call BrainClient.

Prerequisites
- Ollama is running on the MacBook.
- The target model is available in Ollama (default: stanky2).
- A Telegram bot token is created using BotFather.

Install
1. Change into this folder.
2. Install dependencies:
   pip install -r requirements.txt

Run Ollama
1. Start Ollama service.
   On macOS, you can run:
   ollama serve
2. Make sure the model is available:
   ollama list

Run Telegram Bot (Polling)
1. Set environment variables.
   macOS/Linux:
   export TELEGRAM_BOT_TOKEN="YOUR_TOKEN"
   export OLLAMA_CHAT_URL="http://127.0.0.1:11434/api/chat"
   export OLLAMA_MODEL="stanky3"

   Windows PowerShell:
   $env:TELEGRAM_BOT_TOKEN="YOUR_TOKEN"
   $env:OLLAMA_CHAT_URL="http://127.0.0.1:11434/api/chat"
   $env:OLLAMA_MODEL="stanky3"

2. Start bot:
   python telegram_bot.py

Interact from Telegram
1. Open Telegram on your phone or desktop.
2. Search for your bot username (the one created in BotFather).
3. Open the bot chat and press Start or send /start.
4. Send text messages and receive single-message replies.
5. Use /reset to clear only that chat's in-memory history.

Telegram Behavior
- Text messages are sent to Ollama with per-chat history.
- Non-text messages are ignored in V1.
- /start initializes history with the same system prompt as Pi.
- /reset clears that chat history.
- If Ollama is unavailable, user receives:
  Model server is currently unavailable. Please try again later.

Notes
- Chat memory is in-memory only and resets on process restart.
- This bot is a lightweight fallback demo interface and does not replace embodied features.
