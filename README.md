# Reddit Story Scraper Tool

A Flask-based web application that scrapes story content from Reddit. The tool can scrape individual posts and subreddit listings, saving the content to text files.

## Features

- Scrape individual Reddit posts by URL
- Scrape top posts from subreddits with sorting options (hot, top-all, top-week, top-month)
- **Automatic TikTok hook generation** using local LLaMA model (suspenseful hooks under 15 words)
- Save content to .txt files with title, hook, author, and content
- Clean up generated files with one click

## Setup

1. Install Python 3.12 (no virtual environment needed) [[memory:2634148]]
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### LLaMA Model Setup (Optional - for TikTok Hook Generation)

To enable automatic TikTok hook generation using a local LLaMA model:

1. Download a quantized LLaMA model file (GGUF format) from Hugging Face
   - Recommended: `llama-2-7b-chat.Q4_K_M.gguf` or similar quantized model
   - Example sources: https://huggingface.co/TheBloke

2. Place the model file in the project directory and rename it to `model.gguf`

3. The application will automatically detect and load the model on startup

**Note:** Without a model file, the application will still work but hooks will show placeholder text.

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Use the web interface to:
   - Scrape individual posts by URL
   - Scrape top posts from subreddits
   - Download generated .txt files
   - Clean up generated files using the cleanup button

## Cleanup

There are two ways to clean up generated files:

1. Use the "Clean Up All Generated Files" button in the web interface
2. Run the cleanup script directly:
```powershell
.\cleanup.ps1
```

The cleanup process removes all generated .txt files while preserving essential files like README.md, requirements.txt, and app.py. 