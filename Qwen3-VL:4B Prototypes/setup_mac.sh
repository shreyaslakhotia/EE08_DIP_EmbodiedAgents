#!/bin/bash
brew install ffmpeg portaudio  # System drivers
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "Done"
