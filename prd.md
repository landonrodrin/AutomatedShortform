# Reddit Story Scraper Tool — PRD

## Overview

This is a Python 3.12 Flask-based web application that scrapes story content from Reddit. The tool has two primary functions:

1. **Scrape a Reddit Post by URL:**  
   Given a Reddit post URL, extract:
   - Post title
   - Author username
   - Post body text (if available)

   Save this information to a `.txt` file in a clean, readable format.

2. **Scrape Top Post URLs from a Subreddit:**  
   Given a subreddit name (e.g., `r/nosleep` or `r/confessions`), fetch the top post URLs sorted by 'hot' or 'top' order and save them to a `.txt` file, ordered from top to bottom.

---

## Goals

- Build a local tool that scrapes and extracts Reddit post information reliably.
- Save output exclusively to `.txt` files in structured, readable format.
- Use API-based scraping (`praw`) with fallback HTML scraping if necessary.
- Follow clean, maintainable, and environment-aware coding practices.

---

## Non-Goals

- No scraping of Reddit comments or media attachments.
- No user login/authentication.
- No bulk scraping from multiple URLs/subreddits at once.
- No public hosting — this is for local development use.

---

## Technical Requirements

### Environment
- Python 3.12+
- Local Flask server (no deployment)
- Optional front-end HTML form interface (for URL or subreddit input)

### Input
- Reddit post URL (e.g. `https://www.reddit.com/r/nosleep/comments/abc123/story_title`)
- Subreddit name (e.g. `nosleep`, `confessions`)

### Output
- **`.txt` files containing:**
  - For post scraping: title, author, content
  - For subreddit scraping: list of top post URLs in ranked order

---

## API Endpoints

### `POST /scrape_post`
**Description:** Accepts a Reddit post URL and scrapes its content.

**Input:**
- `url` (form-data or JSON string)

**Output:**
- Saves `scraped_post.txt` file containing:

Title: [Post Title]
Author: [Username]
Content:
[Post Body]

- Returns JSON confirmation:
```json
{
  "message": "Post scraped and saved to text file!"
}
```

---

### `POST /scrape_subreddit`
**Description:** Fetches the top 10 (default) post URLs from a subreddit.

**Input:**
- `subreddit` (form-data or JSON string)

**Output:**
- Saves `top_posts_<subreddit>.txt` containing:

1. https://www.reddit.com/r/nosleep/comments/xxx/post1
2. https://www.reddit.com/r/nosleep/comments/yyy/post2

- Returns JSON confirmation:
```json
{
  "message": "Top posts scraped and saved to text file!"
}
```

---

## Rules & Code Constraints

- ✅ Always prefer simple, readable solutions.
- 🔁 Avoid duplication of code — check existing logic first.
- 🌐 Code should consider different environments: dev, test, prod.
- 🧠 Only make changes you clearly understand or that are explicitly requested.
- 🔧 Fix bugs using existing patterns before introducing new ones. If introducing a new pattern, remove redundant code afterward.
- 🧼 Keep the codebase clean, modular, and organized.
- 📜 Avoid creating one-off scripts in random files — keep logic inside structured modules.
- 📏 Refactor files that exceed 200–300 lines of code.
- 🧪 Mock data should be used for tests only. Never mock/stub data for development or production logic.
- 🚫 Never hardcode fake data in dev, staging, or production environments.
- 🔐 Never overwrite existing .env files without explicit approval.
- ✅ Validate that the provided Reddit URL or subreddit exists and is accessible before scraping.
- ⬆️ Subreddit scraping must return top posts from 'hot' or 'top' in proper order.
- 🧾 Set appropriate User-Agent headers for all HTTP requests.
- 🔟 Subreddit scraping defaults to fetching 10 posts (configurable).
- ❌ Provide clear error handling for:
    - Invalid URLs
    - Invalid or restricted subreddits
    - Deleted posts
    - API errors and rate limits

---

## Milestones

1. Set up Flask server with /scrape_post and /scrape_subreddit endpoints.
2. Implement post scraping using praw.
3. Implement subreddit scraping using praw.
4. Save results into .txt files in correct format.
5. Add error handling and edge case validation.
6. (Optional) Build simple HTML form for input.
7. Test all routes and input types.

---

## Dependencies

- Flask
- praw
- requests
- beautifulsoup4

To install:

```
pip install Flask praw requests beautifulsoup4
```

---

## File Structure

reddit_scraper/
├── app.py
├── prd.md
├── requirements.txt
├── scraped_post.txt
├── top_posts_<subreddit>.txt
├── templates/
│   └── index.html (optional)

---

## Stretch Goals (Future Enhancements)

- Scrape top-level comments.
- Download media (images, videos, GIFs).
- Support configurable sorting: 'hot', 'top', 'new', 'controversial'.
- Store scrape logs/history.
- Export results to .json or .csv formats.

---

## ✅ Done — one single, organized, complete `prd.md` you can drop into your project root and Cursor will pick up.
