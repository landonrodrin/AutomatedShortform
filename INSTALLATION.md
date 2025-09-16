# Installation Guide - Reddit Story Scraper with LLaMA Integration

## System Requirements

- Python 3.12+ (without virtual environment)
- Windows 10/11 (PowerShell support)
- At least 8GB RAM (for LLaMA model)
- 4-8GB free disk space (for model file)

## Step-by-Step Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask==2.3.3 (Web framework)
- praw==7.7.1 (Reddit API)
- requests==2.31.0 (HTTP requests)
- beautifulsoup4==4.12.2 (HTML parsing)
- llama-cpp-python==0.2.20 (Local LLaMA model support)

### 2. LLaMA Model Setup (Optional but Recommended)

#### Download a Quantized Model

1. Visit a model repository (e.g., https://huggingface.co/TheBloke)
2. Download a quantized GGUF model file. Recommended options:
   - **llama-2-7b-chat.Q4_K_M.gguf** (~4GB, good balance)
   - **llama-2-13b-chat.Q4_K_M.gguf** (~7GB, better quality)
   - **mistral-7b-instruct-v0.1.Q4_K_M.gguf** (~4GB, alternative)

#### Model Placement

3. Place the downloaded `.gguf` file in your project directory
4. Rename it to exactly `model.gguf`

```
redditscraper/
├── app.py
├── model.gguf  ← Your LLaMA model file
├── requirements.txt
└── ...
```

### 3. Verify Installation

Run the application:
```bash
python app.py
```

Check the console output:
- ✅ `[INFO] Loading LLaMA model from model.gguf`
- ✅ `[INFO] LLaMA model loaded successfully`

Or without model:
- ⚠️ `[WARNING] Model file 'model.gguf' not found. Hook generation will be disabled.`

## Troubleshooting

### Common Issues

**1. Import Error: llama_cpp**
```
[WARNING] llama-cpp-python not installed. Hook generation will be disabled.
```
**Solution:** Reinstall with: `pip install llama-cpp-python==0.2.20`

**2. Model Loading Error**
```
[ERROR] Failed to load LLaMA model: [specific error]
```
**Solutions:**
- Ensure model file is named exactly `model.gguf`
- Check available RAM (need 4-8GB free)
- Try a smaller quantized model (Q4_K_M instead of Q8_0)

**3. Memory Issues**
- Close other applications to free RAM
- Use a smaller model variant
- Reduce `n_ctx` parameter in app.py (line ~25)

### Performance Optimization

- **CPU Usage:** Adjust `n_threads` parameter (currently 4)
- **Memory Usage:** Use Q4_K_M quantization for smaller models
- **Speed:** Smaller models generate hooks faster

## File Output Format

With LLaMA integration enabled, scraped files will have this format:

```
Title: [Reddit Post Title]
TikTok Hook: [Generated suspenseful hook under 15 words]
Author: u/username
Content:
[Full post content...]
```

## Model Recommendations by Use Case

| Use Case | Model | File Size | RAM Required |
|----------|-------|-----------|--------------|
| Testing | llama-2-7b-chat.Q4_K_M.gguf | ~4GB | 6GB |
| Production | llama-2-13b-chat.Q4_K_M.gguf | ~7GB | 10GB |
| Low Resource | mistral-7b-instruct-v0.1.Q4_K_M.gguf | ~4GB | 6GB |

All models will generate hooks, but larger models typically produce higher quality results. 