import os
import re
import praw
import json
import time
import threading
import shutil
import uuid
import requests
import logging
import zipfile
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
import glob
import subprocess

# Fix Windows console encoding issues
import sys
import io

# Fix Windows console encoding at startup
if sys.platform == "win32":
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Set console to UTF-8 if possible
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass

# Configure logging with Windows console encoding fix
class WindowsSafeStreamHandler(logging.StreamHandler):
    def __init__(self):
        if sys.platform == "win32":
            # Use a safe stream for Windows
            try:
                stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            except:
                stream = sys.stdout
        else:
            stream = sys.stdout
        super().__init__(stream)

    def emit(self, record):
        try:
            super().emit(record)
        except (UnicodeEncodeError, OSError):
            # Fallback for encoding issues
            try:
                msg = self.format(record)
                # Strip any problematic characters
                msg = msg.encode('ascii', errors='replace').decode('ascii')
                print(msg)
            except:
                pass

logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO to reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),  # Added UTF-8 encoding
        WindowsSafeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose TTS logging
logging.getLogger('TTS').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.getLogger('TTS.tts.utils.text.phonemizers').setLevel(logging.WARNING)

# TTS imports - Coqui VCTK VITS only
try:
    from TTS.api import TTS
    import torch  # For GPU acceleration
    TTS_AVAILABLE = True
    logger.info("Coqui TTS library available")
except ImportError:
    logger.error("Coqui TTS library not installed. Install with: pip install TTS")
    TTS_AVAILABLE = False

# Video processing imports - ffmpeg
try:
    import ffmpeg
    from pydub import AudioSegment
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    VIDEO_PROCESSING_AVAILABLE = True
    logger.info("ffmpeg and video processing libraries available")
except ImportError:
    logger.warning("Video processing libraries not installed. Video generation will be disabled.")
    VIDEO_PROCESSING_AVAILABLE = False

# LLaMA model imports
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
    logger.info("LLaMA library available")
except ImportError:
    logger.warning("llama-cpp-python not installed. Hook generation will be disabled.")
    LLAMA_AVAILABLE = False

# Whisper imports for subtitle generation
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("Whisper library available")
except ImportError:
    logger.warning("OpenAI Whisper not installed. Subtitle generation will be disabled.")
    WHISPER_AVAILABLE = False

app = Flask(__name__)

# Global variables
llm = None
tts = None
whisper_model = None
processing_queue = {}
video_queue = []  # Queue for video processing jobs
settings = {
    'defaultUsername': 'OminousStories',
    'defaultGameplay': 'minecraft.mp4',
    'defaultMusic': 'simple.mp3',
    'defaultAvatar': 'OminousStoriesLogo.png',
    'outputLocation': 'downloads',
    'batchSize': 'auto'
}

logger.info("Starting Reddit TikTok Video Generator Application")

# Initialize models
if LLAMA_AVAILABLE:
    try:
        model_path = "model.gguf"
        if os.path.exists(model_path):
            logger.info(f"Loading LLaMA model from {model_path}")
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            logger.info("LLaMA model loaded successfully")
        else:
            logger.warning(f"Model file '{model_path}' not found. Hook generation will be disabled.")
    except Exception as e:
        logger.error(f"Failed to load LLaMA model: {e}")
        llm = None

if TTS_AVAILABLE:
    try:
        logger.info("Loading Coqui TTS VCTK VITS model...")
        
        # Check if CUDA is available for GPU acceleration
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing TTS on device: {device}")
            tts = TTS("tts_models/en/vctk/vits", progress_bar=False).to(device)
            logger.info(f"Coqui TTS VCTK VITS model loaded successfully on {device}")
        except Exception as gpu_error:
            logger.warning(f"GPU initialization failed, falling back to CPU: {gpu_error}")
            tts = TTS("tts_models/en/vctk/vits", progress_bar=False)
            logger.info("Coqui TTS VCTK VITS model loaded successfully on CPU")
            
    except Exception as e:
        logger.error(f"Failed to load Coqui TTS model: {e}")
        tts = None

if WHISPER_AVAILABLE:
    try:
        logger.info("Loading Whisper medium model for subtitle generation...")
        
        # Check if CUDA is available for GPU acceleration
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Whisper on device: {device}")
        
        whisper_model = whisper.load_model("medium", device=device)
        logger.info(f"Whisper medium model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        whisper_model = None

# Reddit API
logger.info("Initializing Reddit API connection...")
reddit = praw.Reddit(
    client_id="GAxbB9ODoitQo_0pJe143A",
    client_secret="lXZ0nrpVENiMsxR1inZ6AJo1CABAVw",
    user_agent='RedditStoryScraperTool/1.0 by Landon',
    username="USERNAME",
    password="PASSWORD"
)
logger.info("Reddit API initialized")

# File upload configuration
UPLOAD_FOLDER = '.'
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB
ALLOWED_EXTENSIONS = {
    'gameplay': {'mp4'},
    'music': {'mp3'},
    'avatar': {'png', 'jpg', 'jpeg'}
}

def allowed_file(filename, file_type):
    logger.debug(f"Checking file: {filename} for type: {file_type}")
    result = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]
    logger.debug(f"File check result: {result}")
    return result

def ensure_folders():
    """Ensure all required folders exist"""
    logger.info("Ensuring required folders exist...")
    folders = ['gameplay_videos', 'background_music', 'avatar_images', 'output']
    for folder in folders:
        if not os.path.exists(folder):
            logger.info(f"Creating folder: {folder}")
            os.makedirs(folder)
        else:
            logger.debug(f"Folder exists: {folder}")

def save_settings_to_file():
    """Save settings to JSON file"""
    logger.info("Saving settings to file...")
    try:
        with open('settings.json', 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info("Settings saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return False

def load_settings_from_file():
    """Load settings from JSON file"""
    logger.info("Loading settings from file...")
    global settings
    try:
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                loaded_settings = json.load(f)
                settings.update(loaded_settings)
            logger.info("Settings loaded successfully")
        else:
            logger.info("No settings file found, using defaults")
        return True
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return False

def get_file_info(filepath):
    """Get file information including size and duration"""
    try:
        size = os.path.getsize(filepath)
        size_mb = round(size / (1024 * 1024), 2)
        
        # Get duration for video/audio files
        duration = None
        if filepath.lower().endswith(('.mp4', '.mp3')):
            try:
                if VIDEO_PROCESSING_AVAILABLE:
                    # Use ffmpeg to get duration
                    probe = ffmpeg.probe(filepath)
                    duration_seconds = float(probe['format']['duration'])
                    minutes = int(duration_seconds // 60)
                    seconds = int(duration_seconds % 60)
                    duration = f"{minutes}:{seconds:02d}"
            except Exception as e:
                print(f"[WARNING] Could not get duration for {filepath}: {e}")
        
        return {
            'name': os.path.basename(filepath),
            'size': f"{size_mb} MB",
            'duration': duration
        }
    except Exception as e:
        print(f"[ERROR] Failed to get file info for {filepath}: {e}")
        return None

def get_default_media_file(folder, default_filename):
    """Get default media file with fallback to first available file"""
    try:
        # Ensure folder exists
        if not os.path.exists(folder):
            logger.warning(f"[WARNING] Media folder '{folder}' does not exist")
            return None
        
        # Check if default file exists
        default_path = os.path.join(folder, default_filename)
        if os.path.exists(default_path):
            logger.info(f"[INFO] Using default media file: {default_filename}")
            return default_filename
        
        # Fallback: use first available file in folder
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        
        # Filter by appropriate extensions
        if folder == 'gameplay_videos':
            files = [f for f in files if f.lower().endswith('.mp4')]
        elif folder == 'background_music':
            files = [f for f in files if f.lower().endswith('.mp3')]
        elif folder == 'avatar_images':
            files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if files:
            fallback_file = files[0]
            logger.warning(f"[WARNING] Default '{default_filename}' not found, using fallback: {fallback_file}")
            return fallback_file
        else:
            logger.error(f"[ERROR] No suitable files found in '{folder}' folder")
            return None
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to get media file from '{folder}': {e}")
        return None

def generate_reddit_post_image(hook_text, username, avatar_path=None):
    """Generate Reddit post image using PIL"""
    try:
        logger.info(f"[INFO] Generating Reddit post image for: {hook_text[:50]}...")
        
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.error("[ERROR] Video processing not available - cannot generate Reddit post image")
            return None
            
        # Create Reddit-style post image (448px width, dynamic height)
        width = 448
        
        # Create transparent background with white card
        card_color = (255, 255, 255)  # White card
        
        # Try to load fonts (fallback to default if not available)
        try:
            title_font = ImageFont.truetype("arial.ttf", 18)
            meta_font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            meta_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Calculate text dimensions for dynamic height
        card_padding = 12
        content_left_margin = 12   # Left margin within card
        content_right_margin = 12  # Right margin within card
        content_width = width - (card_padding * 2) - content_left_margin - content_right_margin
        
        logger.info(f"[DEBUG] Title wrapping - Template width: {width}px, Available content width: {content_width}px")
        
        # Wrap title text to calculate height needed
        title_lines = []
        words = hook_text.split()
        current_line = ""
        
        # Create temporary image to measure text
        temp_img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = temp_draw.textbbox((0, 0), test_line, font=title_font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= content_width:
                current_line = test_line
            else:
                if current_line:
                    title_lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long - force it anyway to prevent infinite loop
                    current_line = word
        
        if current_line:
            title_lines.append(current_line)
        
        logger.info(f"[DEBUG] Original text: '{hook_text}'")
        logger.info(f"[DEBUG] Wrapped into {len(title_lines)} lines: {title_lines}")
        
        # Calculate required height
        avatar_size = 32
        line_height = 22
        title_height = len(title_lines) * line_height
        engagement_height = 40
        
        # Total height calculation with extra padding to ensure nothing gets cut off
        height = (
            20 +  # Top padding
            avatar_size + 12 +  # Avatar and spacing
            title_height + 20 +  # Title and extra spacing
            engagement_height +  # Engagement bar
            25  # Bottom padding (increased)
        )
        
        logger.info(f"[DEBUG] Image dimensions: {width}x{height}px, Title lines: {len(title_lines)}, Title height: {title_height}px")
        
        # Create the actual image with transparent background
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw rounded rectangle card
        corner_radius = 12
        card_bbox = [card_padding, card_padding, width - card_padding, height - card_padding]
        draw.rounded_rectangle(card_bbox, radius=corner_radius, fill=card_color, outline=(220, 220, 220), width=1)
        
        # Colors for light theme
        text_color = (28, 28, 28)  # Dark text
        meta_color = (120, 124, 126)  # Gray meta text
        bubble_text_color = (60, 60, 60)  # Dark text for bubble content (like icons)
        upvote_color = (255, 69, 0)   # Reddit upvote orange
        
        # Content area within card
        content_x = card_padding + 12
        y_pos = card_padding + 12
        
        # Load and draw avatar
        avatar_x = content_x
        avatar_y = y_pos
        
        try:
            if avatar_path and os.path.exists(f"avatar_images/{avatar_path}"):
                avatar = Image.open(f"avatar_images/{avatar_path}").convert('RGBA')
                avatar = avatar.resize((avatar_size, avatar_size), Image.Resampling.LANCZOS)
                
                # Create circular mask for avatar
                mask = Image.new('L', (avatar_size, avatar_size), 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.ellipse((0, 0, avatar_size, avatar_size), fill=255)
                
                # Apply mask to avatar
                avatar.putalpha(mask)
                image.paste(avatar, (avatar_x, avatar_y), avatar)
            else:
                # Default avatar circle
                draw.ellipse([avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size], 
                           fill=(99, 99, 99), outline=(170, 170, 170), width=2)
                # Add user icon
                draw.text((avatar_x + avatar_size//2 - 4, avatar_y + avatar_size//2 - 6), "👤", 
                         font=small_font, fill=(255, 255, 255))
        except Exception as e:
            logger.warning(f"Failed to load avatar: {e}")
            # Default avatar circle
            draw.ellipse([avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size], 
                       fill=(99, 99, 99), outline=(170, 170, 170), width=2)
        
        # Draw username with verification badge
        username_x = avatar_x + avatar_size + 8
        username_y = avatar_y + (avatar_size - 14) // 2
        
        # Draw username
        draw.text((username_x, username_y), username, font=meta_font, fill=text_color)
        
        # Get username width for verification badge placement
        username_bbox = draw.textbbox((0, 0), username, font=meta_font)
        username_width = username_bbox[2] - username_bbox[0]
        
        # Draw verification badge (blue checkmark)
        verification_x = username_x + username_width + 4
        verification_y = username_y
        verification_size = 14
        
        # Blue circle for verification
        draw.ellipse([verification_x, verification_y, verification_x + verification_size, verification_y + verification_size], 
                   fill=(29, 161, 242), outline=(29, 161, 242))
        
        # White checkmark
        check_points = [
            (verification_x + 3, verification_y + 7),
            (verification_x + 6, verification_y + 10),
            (verification_x + 11, verification_y + 4)
        ]
        draw.line(check_points[:2], fill=(255, 255, 255), width=2)
        draw.line(check_points[1:], fill=(255, 255, 255), width=2)
        
        # Draw time stamp
        time_x = verification_x + verification_size + 6
        draw.text((time_x, username_y), "• 2h", font=small_font, fill=meta_color)
        
        # Move to title position
        y_pos = avatar_y + avatar_size + 12
        
        # Draw title lines
        logger.info(f"[DEBUG] Drawing {len(title_lines)} title lines starting at y={y_pos}")
        for i, line in enumerate(title_lines):
            logger.info(f"[DEBUG] Drawing line {i+1}: '{line}' at y={y_pos}")
            draw.text((content_x, y_pos), line, font=title_font, fill=text_color)
            y_pos += line_height
        
        # Draw engagement bar
        engagement_y = y_pos + 12
        
        # Load and resize icons with proper transparency handling
        icon_size = 20
        bubble_height = 32
        item_spacing = 12
        padding = 8  # Equal padding around all elements
        
        try:
            # Load PNG icons with proper RGBA conversion
            up_icon = Image.open("icons/up_icon.png").convert('RGBA').resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            bubble_icon = Image.open("icons/bubble_icon.png").convert('RGBA').resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            ribbon_icon = Image.open("icons/ribbon_icon.png").convert('RGBA').resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            send_icon = Image.open("icons/send_icon.png").convert('RGBA').resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            
            current_x = content_x
            
            # Calculate text widths for proper bubble sizing
            upvote_text_width = draw.textbbox((0, 0), "1.2k", font=small_font)[2]
            comment_text_width = draw.textbbox((0, 0), "47", font=small_font)[2]
            trophy_text_width = draw.textbbox((0, 0), "3", font=small_font)[2]
            share_text_width = draw.textbbox((0, 0), "Share", font=small_font)[2]
            
            # Upvote section
            upvote_bubble_width = icon_size + upvote_text_width + (padding * 3)  # icon + text + 3 paddings
            bubble_rect = [current_x, engagement_y, current_x + upvote_bubble_width, engagement_y + bubble_height]
            draw.rounded_rectangle(bubble_rect, radius=16, fill=(245, 245, 245), outline=(220, 220, 220), width=1)
            icon_x = current_x + padding
            icon_y = engagement_y + (bubble_height - icon_size) // 2
            image.paste(up_icon, (icon_x, icon_y), up_icon)
            
            # Upvote count inside bubble
            text_x = icon_x + icon_size + padding
            text_y = engagement_y + (bubble_height - 12) // 2
            draw.text((text_x, text_y), "1.2k", font=small_font, fill=bubble_text_color)
            current_x += upvote_bubble_width + item_spacing
            
            # Comment section
            comment_bubble_width = icon_size + comment_text_width + (padding * 3)
            bubble_rect = [current_x, engagement_y, current_x + comment_bubble_width, engagement_y + bubble_height]
            draw.rounded_rectangle(bubble_rect, radius=16, fill=(245, 245, 245), outline=(220, 220, 220), width=1)
            icon_x = current_x + padding
            icon_y = engagement_y + (bubble_height - icon_size) // 2
            image.paste(bubble_icon, (icon_x, icon_y), bubble_icon)
            
            # Comment count inside bubble
            text_x = icon_x + icon_size + padding
            text_y = engagement_y + (bubble_height - 12) // 2
            draw.text((text_x, text_y), "47", font=small_font, fill=bubble_text_color)
            current_x += comment_bubble_width + item_spacing
            
            # Trophy section
            trophy_bubble_width = icon_size + trophy_text_width + (padding * 3)
            bubble_rect = [current_x, engagement_y, current_x + trophy_bubble_width, engagement_y + bubble_height]
            draw.rounded_rectangle(bubble_rect, radius=16, fill=(245, 245, 245), outline=(220, 220, 220), width=1)
            icon_x = current_x + padding
            icon_y = engagement_y + (bubble_height - icon_size) // 2
            image.paste(ribbon_icon, (icon_x, icon_y), ribbon_icon)
            
            # Trophy count inside bubble
            text_x = icon_x + icon_size + padding
            text_y = engagement_y + (bubble_height - 12) // 2
            draw.text((text_x, text_y), "3", font=small_font, fill=bubble_text_color)
            current_x += trophy_bubble_width + item_spacing
            
            # Share section
            share_bubble_width = icon_size + share_text_width + (padding * 3)
            bubble_rect = [current_x, engagement_y, current_x + share_bubble_width, engagement_y + bubble_height]
            draw.rounded_rectangle(bubble_rect, radius=16, fill=(245, 245, 245), outline=(220, 220, 220), width=1)
            icon_x = current_x + padding
            icon_y = engagement_y + (bubble_height - icon_size) // 2
            image.paste(send_icon, (icon_x, icon_y), send_icon)
            
            # Share text inside bubble
            text_x = icon_x + icon_size + padding
            text_y = engagement_y + (bubble_height - 12) // 2
            draw.text((text_x, text_y), "Share", font=small_font, fill=bubble_text_color)
            
        except Exception as icon_error:
            logger.warning(f"[WARNING] Failed to load icons: {icon_error}")
            # Fallback to text-based engagement bar
            draw.text((content_x, engagement_y), "↑ 1.2k", font=small_font, fill=bubble_text_color)
            draw.text((content_x + 45, engagement_y), "💬 47", font=small_font, fill=bubble_text_color)
            draw.text((content_x + 85, engagement_y), "🏆 3", font=small_font, fill=bubble_text_color)
            draw.text((content_x + 120, engagement_y), "↗ Share", font=small_font, fill=bubble_text_color)
        
        # Save image
        output_path = f"reddit_post_{int(time.time())}.png"
        image.save(output_path)
        logger.info(f"[INFO] Reddit post image saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate Reddit post image: {e}")
        return None



def chunk_text(text, max_words=500):
    """Split text into chunks of approximately max_words without cutting sentences"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_words = len(sentence.split())
        
        if current_word_count + sentence_words > max_words and current_chunk:
            chunks.append(current_chunk.strip() + '.')
            current_chunk = sentence
            current_word_count = sentence_words
        else:
            if current_chunk:
                current_chunk += '. ' + sentence
            else:
                current_chunk = sentence
            current_word_count += sentence_words
    
    if current_chunk:
        chunks.append(current_chunk.strip() + '.')
    
    return chunks

def clean_text_for_tts(text):
    """Clean text for TTS by removing problematic characters"""
    # Remove asterisks and other problematic characters
    text = text.replace('*', '')  # Remove asterisks
    text = text.replace('_', '')  # Remove underscores (italic markdown)
    text = text.replace('~', '')  # Remove tildes (strikethrough markdown)
    text = text.replace('`', '')  # Remove backticks (code markdown)
    text = text.replace('#', '')  # Remove hash symbols (header markdown)
    
    # Clean up extra spaces
    text = ' '.join(text.split())
    
    return text

def generate_tts_audio(text, voice="p232", output_path="output.wav"):
    """Generate TTS audio using Coqui VCTK VITS"""
    try:
        if not tts:
            print("[ERROR] Coqui TTS model not available")
            return False
        
        # Clean text for TTS
        cleaned_text = clean_text_for_tts(text)
        logger.info(f"[INFO] Generating VCTK VITS audio for: {cleaned_text[:50]}...")
        
        # Use Coqui TTS with VCTK VITS model
        tts.tts_to_file(text=cleaned_text, speaker=voice, file_path=output_path)
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate VCTK VITS audio: {e}")
        return False

def combine_audio_files(hook_audio, story_by_audio, content_audio_list, output_path):
    """Combine audio files seamlessly without pauses using pydub and create subtitle audio"""
    try:
        if not VIDEO_PROCESSING_AVAILABLE:
            print("[ERROR] Audio processing not available")
            return False, None
            
        logger.info(f"[INFO] Combining audio files seamlessly...")
        
        # Load audio segments
        hook_segment = AudioSegment.from_wav(hook_audio)
        story_by_segment = AudioSegment.from_wav(story_by_audio)
        
        # Create 0.8 second silence for breathing room between segments
        silence_duration = 800  # 800ms = 0.8 seconds
        silence_segment = AudioSegment.silent(duration=silence_duration)
        
        # Start with hook + 0.5s silence + story_by
        combined = hook_segment + silence_segment + story_by_segment
        
        # Create subtitle audio (content only, excluding hook and story_by)
        subtitle_audio = None
        
        # Add content chunks seamlessly
        for content_audio in content_audio_list:
            if os.path.exists(content_audio):
                content_segment = AudioSegment.from_wav(content_audio)
                combined += content_segment
                
                # Build subtitle audio with only content chunks
                if subtitle_audio is None:
                    subtitle_audio = content_segment
                else:
                    subtitle_audio = subtitle_audio + content_segment
        
        # Export combined audio
        combined.export(output_path, format="wav")
        logger.info(f"[INFO] Seamless audio saved: {output_path}")
        
        # Export subtitle audio for Whisper processing
        subtitle_audio_path = output_path.replace(".wav", "_subtitle.wav")
        if subtitle_audio is not None:
            subtitle_audio.export(subtitle_audio_path, format="wav")
            logger.info(f"[INFO] Subtitle audio saved: {subtitle_audio_path}")
        else:
            logger.warning("[WARNING] No content audio for subtitles - skipping subtitle audio creation")
            subtitle_audio_path = None
        
        # Calculate timing information
        hook_duration = len(hook_segment) / 1000.0
        story_by_duration = len(story_by_segment) / 1000.0
        silence_duration_seconds = silence_duration / 1000.0  # Convert to seconds
        
        # Calculate subtitle start offset: subtitles start after story by audio finishes completely
        # Formula: hook_duration + silence_duration + story_by_duration (1.3x speed adjustment applied later)
        subtitle_start_offset = hook_duration + silence_duration_seconds + story_by_duration
        
        timing_info = {
            'hook_duration': hook_duration,
            'story_by_duration': story_by_duration,
            'silence_duration': silence_duration_seconds,
            'total_duration': len(combined) / 1000.0,
            'subtitle_start_offset': subtitle_start_offset,
            'subtitle_audio_path': subtitle_audio_path
        }
        
        logger.info(f"[INFO] Timing info: hook={timing_info['hook_duration']:.2f}s, "
                   f"silence={timing_info['silence_duration']:.2f}s, "
                   f"story_by={timing_info['story_by_duration']:.2f}s, "
                   f"total={timing_info['total_duration']:.2f}s")
        logger.info(f"[INFO] Subtitle start offset: {subtitle_start_offset:.2f}s (calculated from hook+0.8s_silence+story_by, 1.3x speed adjustment applied later)")
        
        return True, timing_info
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to combine audio files: {e}")
        return False, None

def generate_final_video(reddit_image, audio_path, gameplay_video=None, background_music=None, output_path="output.mp4", hook_duration=None, subtitle_path=None, story_by_overlay_path=None, story_by_duration=None):
    """Generate final video using ffmpeg with hardware acceleration and optimized performance"""
    try:
        if not VIDEO_PROCESSING_AVAILABLE:
            print("[ERROR] Video processing not available")
            return False
            
        logger.info(f"[INFO] Generating final video with hardware acceleration...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get audio duration
        audio_probe = ffmpeg.probe(audio_path)
        audio_duration = float(audio_probe['format']['duration'])
        
        # Use hook duration passed as parameter (calculated before cleanup)
        if hook_duration is None or hook_duration <= 0:
            hook_duration = 3.0  # Default fallback
        
        # Adjust hook duration for 1.3x speed increase
        original_hook_duration = hook_duration
        hook_duration = hook_duration / 1.3  # Shorter duration due to speed increase
        logger.info(f"[INFO] Hook duration adjusted for 1.3x speed: {original_hook_duration:.2f}s → {hook_duration:.2f}s")
        
        # Detect available hardware acceleration
        hw_encoder = detect_hardware_acceleration()
        logger.info(f"[INFO] Using encoder: {hw_encoder}")
        
        # Background video processing
        if gameplay_video and os.path.exists(f"gameplay_videos/{gameplay_video}"):
            video_path = f"gameplay_videos/{gameplay_video}"
            logger.info(f"[INFO] Using gameplay video: {video_path}")
            
            # Get video duration to calculate loop iterations needed
            try:
                video_probe = ffmpeg.probe(video_path)
                video_duration = float(video_probe['format']['duration'])
                loop_count = max(1, int(audio_duration / video_duration) + 1)
                logger.info(f"[INFO] Video duration: {video_duration:.2f}s, will loop {loop_count} times")
            except Exception as e:
                logger.warning(f"[WARNING] Could not get video duration: {e}, using default loop")
                loop_count = 10
            
            # Use stream_loop for efficiency (like working generate_final_video)
            video_input = ffmpeg.input(video_path, stream_loop=loop_count)
            
            # Optimize video processing with hardware acceleration
            video_stream = optimize_video_processing(video_input, video_probe, hw_encoder)
            video_stream = ffmpeg.filter(video_stream, 'setpts', 'PTS-STARTPTS')  # Reset timestamps
            
            # Explicitly mute background video audio (ensure no audio from background video is used)
            logger.info("[INFO] Background video audio is muted (only using TTS and background music)")
            # Note: We only use video stream, no audio extraction from background video
            
        else:
            # Create solid color background
            logger.info(f"[INFO] Creating solid background")
            video_stream = ffmpeg.input('color=black:size=1080x1920:rate=30', f='lavfi', t=audio_duration)
        
        # SCALED UP Reddit post overlay - NO FADE EFFECTS
        if reddit_image and os.path.exists(reddit_image):
            logger.info(f"[INFO] Adding LARGE Reddit post overlay (NO FADE): {reddit_image}")
            
            overlay_input = ffmpeg.input(reddit_image)
            
            # SCALE UP THE TEMPLATE UNIFORMLY (no stretching or forced aspect ratios)
            # Target: 80% of screen width (864px instead of 448px)
            target_width = 864  # 80% of 1080px
            scale_factor = target_width / 448  # Original template is 448px wide
            
            # Get the ACTUAL dimensions of the generated template image
            temp_img = Image.open(reddit_image)
            original_width, original_height = temp_img.size  # Get real dimensions
            temp_img.close()
            
            # Scale height using the SAME factor (uniform scaling - no stretching)
            target_height = int(original_height * scale_factor)  # Uniform scaling
            
            logger.info(f"[INFO] Uniform scaling: {original_width}x{original_height} → {target_width}x{target_height} (scale: {scale_factor:.2f}x)")
            
            # Scale the overlay to be much larger and more prominent
            scaled_overlay = ffmpeg.filter(overlay_input, 'scale', target_width, target_height)
            
            # Position the larger template in center of screen
            center_x = int((1080 - target_width) / 2)  # Center horizontally with proper rounding
            center_y = int((1920 - target_height) / 2)  # Center vertically
            
            logger.info(f"[INFO] Positioning large template at ({center_x}, {center_y}) - size: {target_width}x{target_height}")
            logger.info(f"[DEBUG] Centering calculation: X=(1080 - {target_width})/2 = {center_x}, Y=(1920 - {target_height})/2 = {center_y}")
            
            # Simple overlay with timing - NO FADE IN/OUT
            video_stream = ffmpeg.overlay(
                video_stream, 
                scaled_overlay, 
                x=center_x,
                y=center_y,
                enable=f'between(t,0,{hook_duration})'  # Show during entire hook duration
            )
            logger.info(f"[INFO] LARGE template will be visible from 0s to {hook_duration:.2f}s (adjusted for 1.3x TTS speed)")
        else:
            logger.warning(f"[WARNING] No Reddit image found, skipping overlay")
        
        # Audio processing
        audio_streams = []
        
        # Main TTS audio - SPEED UP BY 1.3x
        tts_audio = ffmpeg.input(audio_path)
        tts_audio_sped = ffmpeg.filter(tts_audio, 'atempo', 1.3)  # Speed up by 1.3x
        audio_streams.append(tts_audio_sped)
        
        # Calculate new audio duration after speed increase
        original_audio_duration = audio_duration
        audio_duration = audio_duration / 1.3  # Shorter duration due to speed increase
        logger.info(f"[INFO] TTS audio sped up 1.3x: {original_audio_duration:.2f}s → {audio_duration:.2f}s")
        
        # Background music processing (simplified)
        if background_music and os.path.exists(f"background_music/{background_music}"):
            music_path = f"background_music/{background_music}"
            logger.info(f"[INFO] Adding background music: {music_path}")
            
            try:
                music_input = ffmpeg.input(music_path)
                music_stream = ffmpeg.filter(music_input, 'aloop', loop=-1, size=2e+09)  # Loop audio
                music_stream = ffmpeg.filter(music_stream, 'volume', 0.15)  # 15% volume
                music_stream = ffmpeg.filter(music_stream, 'atrim', duration=audio_duration)
                audio_streams.append(music_stream)
            except Exception as e:
                logger.warning(f"Could not process background music: {e}")
        
        # Mix audio streams
        if len(audio_streams) > 1:
            mixed_audio = ffmpeg.filter(audio_streams, 'amix', inputs=len(audio_streams), duration='longest')
        else:
            mixed_audio = audio_streams[0]
        
        # Apply subtitles if available
        if subtitle_path and os.path.exists(subtitle_path):
            logger.info(f"[INFO] Adding subtitles from: {subtitle_path}")
            
            # Convert Windows paths to forward slashes for ffmpeg
            subtitle_path_ffmpeg = subtitle_path.replace('\\', '/')
            video_stream = ffmpeg.filter(video_stream, 'subtitles', subtitle_path_ffmpeg)
        else:
            logger.info("No subtitles to add")
        
        # Add story by text overlay if available
        if story_by_overlay_path and os.path.exists(story_by_overlay_path):
            logger.info(f"[INFO] Adding story by text overlay: {story_by_overlay_path}")
            
            # Validate timing parameters
            if not hook_duration or not story_by_duration:
                logger.error("[ERROR] Missing hook_duration or story_by_duration for overlay timing")
                logger.info("[INFO] No story by text overlay to add")
            else:
                # Calculate timing for story by overlay
                # Start: after hook duration + 0.8s silence + 0.9s delay, End: start + story_by_duration - 0.4s (early disappear)
                # Adjust for 1.3x speed
                silence_duration = 0.8  # 0.8s silence added between hook and story_by
                overlay_start = ((hook_duration + silence_duration) / 1.3) + 0.9
                # Calculate the overlay duration: full story by duration minus 0.4s early disappear
                full_overlay_duration = max(0.1, story_by_duration - 0.35)
                overlay_duration = full_overlay_duration / 1.3  # Adjust for 1.3x speed
                overlay_end = overlay_start + overlay_duration
                
                logger.info(f"[INFO] Story by overlay timing: {overlay_start:.2f}s to {overlay_end:.2f}s (duration: {overlay_duration:.2f}s)")
                logger.info(f"[INFO] Original story by duration: {story_by_duration:.2f}s, adjusted duration: {full_overlay_duration:.2f}s")
                
                # Add overlay with timing
                overlay_input = ffmpeg.input(story_by_overlay_path)
                video_stream = ffmpeg.filter(
                    [video_stream, overlay_input], 
                    'overlay', 
                    enable=f'between(t,{overlay_start},{overlay_end})',
                    x='0', y='0'  # Overlay covers full screen
                )
        else:
            logger.info("[INFO] No story by text overlay to add")
        
        # Build output with hardware acceleration
        output_args = build_output_args(hw_encoder, audio_duration)
        
        output = ffmpeg.output(
            video_stream, mixed_audio, 
            output_path,
            **output_args
        )
        
        # Run ffmpeg with better error handling and timing
        import time
        start_time = time.time()
        
        try:
            logger.info(f"[INFO] Starting ffmpeg encoding with {hw_encoder}...")
            result = ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            encoding_time = time.time() - start_time
            logger.info(f"[INFO] Encoding completed in {encoding_time:.2f} seconds")
            
        except ffmpeg.Error as e:
            logger.error(f"[ERROR] ffmpeg command failed:")
            if e.stdout:
                stdout_text = e.stdout.decode('utf-8', errors='replace')
                logger.error(f"[ERROR] stdout: {stdout_text}")
            if e.stderr:
                stderr_text = e.stderr.decode('utf-8', errors='replace')
                logger.error(f"[ERROR] stderr: {stderr_text}")
                # Check for specific error patterns
                if "Invalid data found" in stderr_text:
                    logger.error("[ERROR] Media file corruption detected")
                elif "No such file" in stderr_text:
                    logger.error("[ERROR] Missing input file")
                elif "Permission denied" in stderr_text:
                    logger.error("[ERROR] File permission issue")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error during encoding: {e}")
            return False
        
        # Validate output file
        if os.path.exists(output_path):
            try:
                # Check file size (should be > 100KB for valid video)
                file_size = os.path.getsize(output_path)
                if file_size < 100000:  # Less than 100KB
                    logger.error(f"[ERROR] Output video suspiciously small: {file_size} bytes - likely corrupted")
                    return False
                
                # Try to probe the output video
                output_probe = ffmpeg.probe(output_path)
                duration = float(output_probe['format']['duration'])
                logger.info(f"[INFO] Output video validated - Size: {file_size/1024/1024:.2f}MB, Duration: {duration:.2f}s")
                logger.info(f"[INFO] Final video generated: {output_path}")
                return True
                
            except Exception as validation_error:
                logger.error(f"[ERROR] Output video validation failed: {validation_error}")
                return False
        else:
            logger.error(f"[ERROR] Video file not created")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate video: {e}")
        return False

def detect_hardware_acceleration():
    """Detect available hardware acceleration and return best encoder"""
    try:
        # Test NVENC (NVIDIA)
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                   capture_output=True, text=True, timeout=5)
            if 'h264_nvenc' in result.stdout:
                logger.info("[INFO] NVENC hardware acceleration detected")
                return 'nvenc'
        except:
            pass
        
        # Test AMD AMF
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                   capture_output=True, text=True, timeout=5)
            if 'h264_amf' in result.stdout:
                logger.info("[INFO] AMD AMF hardware acceleration detected")
                return 'amf'
        except:
            pass
        
        # Test Intel QSV
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                   capture_output=True, text=True, timeout=5)
            if 'h264_qsv' in result.stdout:
                logger.info("[INFO] Intel QSV hardware acceleration detected")
                return 'qsv'
        except:
            pass
        
        logger.info("[INFO] No hardware acceleration detected, using CPU")
        return 'cpu'
        
    except Exception as e:
        logger.warning(f"[WARNING] Hardware detection failed: {e}, using CPU")
        return 'cpu'

def optimize_video_processing(video_input, video_probe, hw_encoder):
    """Optimize video processing based on hardware capabilities"""
    try:
        # Get video dimensions
        video_info = video_probe['streams'][0]
        input_width = int(video_info['width'])
        input_height = int(video_info['height'])
        
        # Target dimensions for TikTok format
        target_width = 1080
        target_height = 1920
        
        # Calculate optimal crop and scale
        target_ratio = target_width / target_height
        input_ratio = input_width / input_height
        
        if input_ratio > target_ratio:
            # Input is wider, crop width
            crop_width = int(input_height * target_ratio)
            crop_height = input_height
            crop_x = (input_width - crop_width) // 2
            crop_y = 0
        else:
            # Input is taller, crop height
            crop_width = input_width
            crop_height = int(input_width / target_ratio)
            crop_x = 0
            crop_y = (input_height - crop_height) // 2
        
        # Apply optimized processing
        if hw_encoder != 'cpu':
            # Use hardware-accelerated scaling when available
            video_stream = ffmpeg.filter(video_input, 'crop', crop_width, crop_height, crop_x, crop_y)
            video_stream = ffmpeg.filter(video_stream, 'scale', target_width, target_height)
        else:
            # CPU fallback
            video_stream = ffmpeg.filter(video_input, 'crop', crop_width, crop_height, crop_x, crop_y)
            video_stream = ffmpeg.filter(video_stream, 'scale', target_width, target_height)
        
        return video_stream
        
    except Exception as e:
        logger.warning(f"[WARNING] Video optimization failed: {e}, using simple scale")
        return ffmpeg.filter(video_input, 'scale', 1080, 1920)

def build_output_args(hw_encoder, audio_duration):
    """Build optimized output arguments based on hardware capabilities"""
    base_args = {
        'acodec': 'aac',
        't': audio_duration,
        'b:a': '128k'
    }
    
    if hw_encoder == 'nvenc':
        return {
            **base_args,
            'vcodec': 'h264_nvenc',
            'preset': 'fast',
            'b:v': '2M',
            'rc': 'vbr',
            'cq': '23'
        }
    elif hw_encoder == 'amf':
        return {
            **base_args,
            'vcodec': 'h264_amf',
            'quality': 'speed',
            'b:v': '2M',
            'rc': 'vbr_peak'
        }
    elif hw_encoder == 'qsv':
        return {
            **base_args,
            'vcodec': 'h264_qsv',
            'preset': 'fast',
            'b:v': '2M'
        }
    else:
        return {
            **base_args,
            'vcodec': 'libx264',
            'preset': 'fast',
            'b:v': '2M',
            'crf': '23'
        }

def process_video_generation(process_id, post_data, mode, user_options=None):
    """Process video generation in background thread"""
    global processing_queue
    
    logger.info(f"Starting video generation process: {process_id}")
    logger.debug(f"Mode: {mode}, Post data: {post_data}")
    
    # Initialize cleanup variables
    hook_audio_path = None
    story_by_audio_path = None
    chunk_audio_paths = []
    final_audio_path = None
    reddit_image = None
    story_by_overlay_path = None
    
    try:
        processing_queue[process_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting video generation...',
            'details': '',
            'queue': []
        }
        
        # Extract data from post
        title = post_data.get('title', '')
        content = post_data.get('content', '')
        hook = post_data.get('hook', '')
        reddit_author = post_data.get('author', 'u/UnknownUser')  # Actual Reddit author
        
        logger.info(f"Processing: {title[:50]}...")
        logger.info(f"[DEBUG] Reddit author extracted: {reddit_author}")
        logger.debug(f"Content length: {len(content)} chars, Hook: {hook[:50]}...")
        
        # Step 1: Generate Reddit post image (10%)
        processing_queue[process_id]['progress'] = 10
        processing_queue[process_id]['message'] = 'Generating Reddit post overlay...'
        
        # Use user-specified username or default
        username = user_options.get('username', settings.get('defaultUsername', 'OminousStories')) if user_options else settings.get('defaultUsername', 'OminousStories')
        avatar_path = user_options.get('avatar', get_default_media_file('avatar_images', 'OminousStoriesLogo.png')) if user_options else get_default_media_file('avatar_images', 'OminousStoriesLogo.png')
        
        logger.info(f"[DEBUG] Using avatar: {avatar_path} {'(user-selected)' if user_options and user_options.get('avatar') else '(default)'}")
        
        # Validate avatar file exists if specified
        if avatar_path and not os.path.exists(f"avatar_images/{avatar_path}"):
            logger.warning(f"[WARNING] Avatar file not found: {avatar_path}, will use default placeholder")
        
        logger.info(f"[DEBUG] Generating Reddit post image for user: '{username}' (from user options)")
        reddit_image = generate_reddit_post_image(hook, username, avatar_path)
        if not reddit_image:
            logger.error("Failed to generate Reddit post image")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = 'Failed to generate Reddit post image'
            return
        logger.info(f"Reddit post image generated: {reddit_image}")
        
        # Step 2: Generate TTS audio (50%)
        processing_queue[process_id]['progress'] = 30
        processing_queue[process_id]['message'] = 'Generating text-to-speech audio...'
        
        voice = user_options.get('voice', 'p232') if user_options else 'p232'
        logger.info(f"[DEBUG] Using TTS voice: {voice} {'(user-selected)' if user_options and user_options.get('voice') else '(default)'}")
        
        # Generate hook audio
        logger.info("Generating hook audio...")
        hook_audio_path = f"temp_hook_{process_id}.wav"
        if not generate_tts_audio(hook, voice, hook_audio_path):
            logger.error("Failed to generate hook audio")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = 'Failed to generate hook audio'
            return
        logger.info("Hook audio generated")
        
        # Generate "Story by: username" audio
        logger.info("Generating story by audio...")
        logger.info(f"[DEBUG] Using reddit_author for TTS: '{reddit_author}'")
        story_by_text = f"Story by: {reddit_author}"  # Use actual Reddit author with colon
        logger.info(f"[DEBUG] Final story_by_text: '{story_by_text}'")
        story_by_audio_path = f"temp_story_by_{process_id}.wav"
        if not generate_tts_audio(story_by_text, voice, story_by_audio_path):
            logger.error("Failed to generate story by audio")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = 'Failed to generate story by audio'
            return
        logger.info(f"Story by audio generated: {story_by_text}")
        
        # Generate content audio in chunks
        logger.info("Chunking content for TTS...")
        content_chunks = chunk_text(content)
        chunk_audio_paths = []
        logger.info(f"Content split into {len(content_chunks)} chunks")
        
        for i, chunk in enumerate(content_chunks):
            logger.debug(f"Generating audio for chunk {i+1}/{len(content_chunks)}")
            chunk_audio_path = f"temp_chunk_{process_id}_{i}.wav"
            if not generate_tts_audio(chunk, voice, chunk_audio_path):
                logger.error(f"Failed to generate audio for chunk {i+1}")
                processing_queue[process_id]['status'] = 'failed'
                processing_queue[process_id]['message'] = f'Failed to generate audio for chunk {i+1}'
                return
            chunk_audio_paths.append(chunk_audio_path)
        logger.info("All content audio files generated")
        
        # Step 3: Combine audio files (70%)
        processing_queue[process_id]['progress'] = 70
        processing_queue[process_id]['message'] = 'Combining audio files...'
        
        logger.info("Combining audio files...")
        final_audio_path = f"final_audio_{process_id}.wav"
        combine_success, timing_info = combine_audio_files(hook_audio_path, story_by_audio_path, chunk_audio_paths, final_audio_path)
        if not combine_success:
            logger.error("Failed to combine audio files")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = 'Failed to combine audio files'
            return
        logger.info("Audio files combined successfully")
        
        # Calculate hook duration BEFORE cleanup for template timing
        if not timing_info:
            logger.error("No timing information available - cannot proceed with video generation")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = 'No timing information available'
            return
        
        hook_duration = timing_info['hook_duration']
        logger.info(f"[INFO] Hook duration from timing info: {hook_duration:.2f}s")
        
        # Step 3.5: Generate subtitles (75%)
        processing_queue[process_id]['progress'] = 75
        processing_queue[process_id]['message'] = 'Generating subtitles...'
        
        subtitle_path = None
        if timing_info and whisper_model and timing_info['subtitle_audio_path']:
            logger.info("Generating Whisper subtitles...")
            subtitle_segments = generate_whisper_subtitles(
                timing_info['subtitle_audio_path'], 
                timing_info['subtitle_start_offset']
            )
            
            if subtitle_segments:
                # Adjust subtitle timing for 1.3x TTS speed increase
                adjusted_segments = []
                for segment in subtitle_segments:
                    adjusted_segment = {
                        "start": segment["start"] / 1.3,  # Adjust for 1.3x speed
                        "end": segment["end"] / 1.3,      # Adjust for 1.3x speed
                        "text": segment["text"]
                    }
                    adjusted_segments.append(adjusted_segment)
                
                subtitle_path = f"subtitles_{process_id}.ass"
                if create_ass_subtitle_file(adjusted_segments, subtitle_path):
                    logger.info(f"Subtitles generated with 1.3x speed adjustment: {subtitle_path}")
                else:
                    logger.warning("Failed to create subtitle file")
                    subtitle_path = None
            else:
                logger.warning("No subtitle segments generated")
        else:
            if not timing_info:
                logger.info("Skipping subtitle generation (no timing info)")
            elif not whisper_model:
                logger.info("Skipping subtitle generation (Whisper not available)")
            elif not timing_info['subtitle_audio_path']:
                logger.info("Skipping subtitle generation (no content audio for subtitles)")
            else:
                logger.info("Skipping subtitle generation (unknown reason)")
        
        # 🧹 IMMEDIATE CLEANUP: Delete temporary audio files after combination
        temp_audio_files = [hook_audio_path, story_by_audio_path] + chunk_audio_paths
        if timing_info and timing_info['subtitle_audio_path']:
            temp_audio_files.append(timing_info['subtitle_audio_path'])
        safe_cleanup_files(temp_audio_files, "temporary audio files")
        # Clear the paths since files are deleted
        hook_audio_path = story_by_audio_path = None
        chunk_audio_paths = []
        
        # Step 4: Generate video (90%)
        processing_queue[process_id]['progress'] = 90
        processing_queue[process_id]['message'] = 'Generating final video...'
        
        # Get media files for video generation with user options or fallbacks
        if user_options:
            # Get user selections or fallback to defaults
            user_gameplay = user_options.get('gameplay')
            user_music = user_options.get('music')
            
            # Validate user-selected gameplay video
            if user_gameplay and os.path.exists(f"gameplay_videos/{user_gameplay}"):
                gameplay_video = user_gameplay
                logger.info(f"[DEBUG] Using user-selected gameplay: {gameplay_video}")
            else:
                gameplay_video = get_default_media_file('gameplay_videos', 'YoutubeVideo.mp4')
                if user_gameplay:
                    logger.warning(f"[WARNING] User-selected gameplay '{user_gameplay}' not found, using default: {gameplay_video}")
                else:
                    logger.info(f"[DEBUG] No user gameplay selected, using default: {gameplay_video}")
            
            # Validate user-selected background music
            if user_music and os.path.exists(f"background_music/{user_music}"):
                background_music = user_music
                logger.info(f"[DEBUG] Using user-selected music: {background_music}")
            else:
                background_music = get_default_media_file('background_music', 'YoutubeAudio.mp3')
                if user_music:
                    logger.warning(f"[WARNING] User-selected music '{user_music}' not found, using default: {background_music}")
                else:
                    logger.info(f"[DEBUG] No user music selected, using default: {background_music}")
        else:
            gameplay_video = get_default_media_file('gameplay_videos', 'YoutubeVideo.mp4')
            background_music = get_default_media_file('background_music', 'YoutubeAudio.mp3')
            logger.info(f"[DEBUG] Using default media - gameplay: {gameplay_video}, music: {background_music}")
        
        # Final validation before video generation
        if not gameplay_video or not os.path.exists(f"gameplay_videos/{gameplay_video}"):
            logger.error(f"[ERROR] Gameplay video not found: {gameplay_video}")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = f'Gameplay video not found: {gameplay_video}'
            return
            
        if not background_music or not os.path.exists(f"background_music/{background_music}"):
            logger.error(f"[ERROR] Background music not found: {background_music}")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = f'Background music not found: {background_music}'
            return
        
        # Test media files with ffprobe to ensure they're valid
        try:
            gameplay_path = f"gameplay_videos/{gameplay_video}"
            music_path = f"background_music/{background_music}"
            
            # Test gameplay video
            gameplay_probe = ffmpeg.probe(gameplay_path)
            logger.info(f"[DEBUG] Gameplay video validated - Duration: {float(gameplay_probe['format']['duration']):.2f}s")
            
            # Test background music
            music_probe = ffmpeg.probe(music_path)
            logger.info(f"[DEBUG] Background music validated - Duration: {float(music_probe['format']['duration']):.2f}s")
            
        except Exception as probe_error:
            logger.error(f"[ERROR] Media file validation failed: {probe_error}")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = f'Media file validation failed: {str(probe_error)}'
            return
        
        logger.info(f"[INFO] Final video generation with VALIDATED media - gameplay: {gameplay_video}, music: {background_music}")
        
        # Generate story by text overlay
        story_by_overlay_path = f"story_by_overlay_{process_id}.png"
        if not generate_story_by_text_overlay(story_by_text, story_by_overlay_path):
            logger.warning("Failed to generate story by text overlay")
            story_by_overlay_path = None
        
        # Create filename based on first 5 words of title
        safe_title = create_safe_filename(title, max_words=5)
        final_video_path = f"output/{safe_title}.mp4"
        
        # Pass hook_duration, subtitle_path, and story by overlay for video generation
        story_by_duration = timing_info['story_by_duration']
        if not generate_final_video(reddit_image, final_audio_path, gameplay_video, background_music, final_video_path, hook_duration, subtitle_path, story_by_overlay_path, story_by_duration):
            logger.error("Failed to generate final video")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = 'Failed to generate final video'
            return
        logger.info(f"Final video generated: {final_video_path}")
        
        # 🧹 IMMEDIATE CLEANUP: Delete final audio file after video generation
        safe_cleanup_files([final_audio_path], "final audio file")
        final_audio_path = None
        
        # Step 5: Complete (100%)
        processing_queue[process_id]['progress'] = 100
        processing_queue[process_id]['message'] = 'Video generation complete!'
        processing_queue[process_id]['status'] = 'completed'
        processing_queue[process_id]['results'] = {
            'message': 'Video generated successfully!',
            'files': [os.path.basename(final_video_path)]  # Just the filename for download links
        }
        
        logger.info(f"Video generation completed successfully: {process_id}")
        
    except Exception as e:
        logger.error(f"Video generation failed for {process_id}: {str(e)}")
        processing_queue[process_id]['status'] = 'failed'
        processing_queue[process_id]['message'] = f'Video generation failed: {str(e)}'
    
    finally:
        # 🧹 FINAL CLEANUP: Ensure any remaining temp files are cleaned up
        remaining_files = []
        if hook_audio_path:
            remaining_files.append(hook_audio_path)
        if story_by_audio_path:
            remaining_files.append(story_by_audio_path)
        if chunk_audio_paths:
            remaining_files.extend(chunk_audio_paths)
        if final_audio_path:
            remaining_files.append(final_audio_path)
        if reddit_image:
            remaining_files.append(reddit_image)
        if subtitle_path:
            remaining_files.append(subtitle_path)
        if story_by_overlay_path:
            remaining_files.append(story_by_overlay_path)
        
        if remaining_files:
            safe_cleanup_files(remaining_files, "remaining temporary files")

def generate_tiktok_hook(title, content):
    """
    Generate a short, suspenseful TikTok hook using local LLaMA model.
    
    Args:
        title (str): Reddit post title
        content (str): Reddit post content
        
    Returns:
        str: Generated hook under 40 words, or "Could not generate hook" if model unavailable
    """
    if not llm:
        logger.warning("LLaMA model not available for hook generation - using fallback")
        # Fallback: use first sentence of title/content as hook
        fallback_hook = title[:100] if title else content[:100] if content else "An interesting story"
        logger.info(f"[DEBUG] Using fallback hook (no LLaMA): {fallback_hook}")
        return fallback_hook
    
    # Extract key context from content
    content_excerpt = content[:150] if content else ""
    
    # Simple, effective prompt that consistently works
    prompt = f"""Write a suspenseful hook for this story that ends with "until" or "but":

Story: {title}
Context: {content_excerpt}

Hook format: Start with something normal, then end with "until" or "but" revealing something disturbing.

Hook:"""

    try:
        logger.info(f"[DEBUG] Generating hook for: {title[:50]}...")
        
        response = llm(
            prompt,
            max_tokens=100,
            temperature=0.6,
            top_p=0.8,
            stop=["\n", "Story:", "Context:", "Hook:", "Format:"],
            echo=False,
            repeat_penalty=1.1
        )
        
        hook = response['choices'][0]['text'].strip()
        logger.info(f"[DEBUG] Raw LLaMA response: '{hook}'")
        
        # Clean and validate the hook
        hook = clean_and_validate_hook(hook)
        
        if hook and len(hook.split()) >= 5:
            logger.info(f"[DEBUG] Generated hook: {hook}")
            return hook
        else:
            logger.warning(f"[WARNING] Hook validation failed: '{hook}' - using fallback")
            # Fallback: use first sentence of title/content as hook
            fallback_hook = title[:100] if title else content[:100] if content else "An interesting story"
            logger.info(f"[DEBUG] Using fallback hook: {fallback_hook}")
            return fallback_hook
            
    except Exception as e:
        logger.error(f"[ERROR] Hook generation failed: {e}")
        # Fallback: use first sentence of title/content as hook
        fallback_hook = title[:100] if title else content[:100] if content else "An interesting story"
        logger.info(f"[DEBUG] Using fallback hook after error: {fallback_hook}")
        return fallback_hook


def clean_and_validate_hook(raw_hook):
    """
    Clean and validate a generated hook from LLaMA
    
    Args:
        raw_hook (str): Raw hook text from model
        
    Returns:
        str: Cleaned hook or empty string if invalid
    """
    if not raw_hook:
        return ""
    
    # Remove quotes and extra whitespace
    hook = raw_hook.replace('"', '').replace("'", '').strip()
    
    # Remove common prefixes that leak through
    prefixes_to_remove = [
        'hook:', 'answer:', 'response:', 'your hook:', 'format:', 
        'example:', 'write:', 'create:', 'generated:', 'output:',
        'title:', 'story:', 'prompt:', 'text:', 'result:'
    ]
    hook_lower = hook.lower()
    for prefix in prefixes_to_remove:
        if hook_lower.startswith(prefix):
            hook = hook[len(prefix):].strip()
            break
    
    # Remove common suffixes
    suffixes_to_remove = ['...', '..', '.', '!', '?']
    for suffix in suffixes_to_remove:
        if hook.endswith(suffix):
            hook = hook[:-len(suffix)].strip()
    
    # Validate hook quality
    if len(hook) < 10:  # Too short
        return ""
    
    if len(hook) > 200:  # Too long
        return ""
    
    # Check for common failure patterns
    failure_patterns = [
        'i cannot', 'i can\'t', 'as an ai', 'i\'m not able',
        'i don\'t have', 'sorry', 'i apologize', 'i\'m sorry'
    ]
    
    hook_lower = hook.lower()
    for pattern in failure_patterns:
        if pattern in hook_lower:
            return ""
    
    # Ensure it's a proper hook (has some story elements)
    if not any(word in hook_lower for word in ['i', 'my', 'the', 'was', 'were', 'had', 'thought', 'until', 'but']):
        return ""
    
    # Add proper ending punctuation if missing
    if not hook.endswith(('.', '!', '?', '...')):
        hook += '.'
    
    return hook

def get_top_comments(submission):
    """Get top 10 comments from a Reddit submission with filtering"""
    try:
        # Ensure comments are loaded
        submission.comments.replace_more(limit=0)
        
        # Get all top-level comments
        top_level_comments = [comment for comment in submission.comments if hasattr(comment, 'body')]
        
        # Filter comments
        filtered_comments = []
        for comment in top_level_comments:
            # Skip deleted/removed comments
            if comment.body in ['[deleted]', '[removed]']:
                continue
            
            # Skip comments that are just links or have URLs
            if 'http' in comment.body or 'www.' in comment.body:
                continue
            
            # Filter out content after "edit:" (case insensitive)
            comment_body = comment.body
            edit_index = comment_body.lower().find('edit:')
            if edit_index != -1:
                comment_body = comment_body[:edit_index].strip()
            
            # Skip if comment is empty after edit filtering
            if not comment_body.strip():
                continue
            
            # Check word count (5-500 words)
            word_count = len(comment_body.split())
            if word_count < 5 or word_count > 500:
                continue
            
            filtered_comments.append({
                'body': comment_body,
                'score': comment.score,
                'author': comment.author.name if comment.author else '[deleted]'
            })
        
        # Sort by score (highest first) and take top 10
        filtered_comments.sort(key=lambda x: x['score'], reverse=True)
        return filtered_comments[:10]
        
    except Exception as e:
        logger.error(f"Error getting top comments: {e}")
        return []

def generate_comment_audio_with_pause(number_text, comment_body, voice, output_path):
    """Generate comment audio with 0.3s pause between number and content for better subtitle segmentation"""
    try:
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.error("Audio processing not available")
            return False
        
        # Generate separate audio for number and comment content
        
        # Create temporary files for number and content audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as number_temp:
            number_audio_path = number_temp.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as content_temp:
            content_audio_path = content_temp.name
        
        # Generate TTS for number part (e.g., "1.")
        if not generate_tts_audio(number_text, voice, number_audio_path):
            logger.error(f"Failed to generate number audio: {number_text}")
            return False
        
        # Generate TTS for comment content
        if not generate_tts_audio(comment_body, voice, content_audio_path):
            logger.error(f"Failed to generate content audio: {comment_body[:50]}...")
            return False
        
        # Load audio segments
        number_segment = AudioSegment.from_wav(number_audio_path)
        content_segment = AudioSegment.from_wav(content_audio_path)
        
        # Create 0.3 second pause for Whisper to detect separate segments
        pause_segment = AudioSegment.silent(duration=300)  # 300ms = 0.3 seconds
        
        # Combine: number + pause + content
        combined = number_segment + pause_segment + content_segment
        
        # Export combined audio
        combined.export(output_path, format="wav")
        
        # Cleanup temporary files
        try:
            os.unlink(number_audio_path)
            os.unlink(content_audio_path)
        except:
            pass  # Ignore cleanup errors
        
        logger.info(f"Generated comment audio with 0.3s pause: '{number_text}' -> [pause] -> '{comment_body[:50]}...'")
        return True
        
    except Exception as e:
        logger.error(f"Error generating comment audio with pause: {e}")
        return False

def process_comment_compilation(process_id, post_data, mode, user_options=None):
    """Process comment compilation video generation"""
    try:
        processing_queue[process_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting comment compilation...',
            'details': '',
            'queue': []
        }
        
        title = post_data.get('title', '')
        comments = post_data.get('comments', [])
        
        logger.info(f"Processing comment compilation: {title[:50]}...")
        
        # Step 1: Generate template intro (20%)
        processing_queue[process_id]['progress'] = 20
        processing_queue[process_id]['message'] = 'Generating template intro...'
        
        # Use original title for template (no LLaMA generation)
        template_text = title
        
        # Generate template audio
        template_audio_path = f"template_audio_{process_id}.wav"
        voice = user_options.get('voice', 'p232') if user_options else 'p232'
        
        if not generate_tts_audio(template_text, voice, template_audio_path):
            logger.error("Failed to generate template audio")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = 'Failed to generate template audio'
            return
        
        # Step 2: Generate numbered comments audio with pauses (60%)
        processing_queue[process_id]['progress'] = 60
        processing_queue[process_id]['message'] = 'Generating comment audio with pauses...'
        
        # Generate individual comment audio files with 0.3s pause after numbers
        comment_audio_paths = []
        for i, comment in enumerate(comments, 1):
            comment_audio_path = f"comment_{i+1}_{process_id}.wav"
            
            # Generate comment with pause between number and content for better subtitle segmentation
            if not generate_comment_audio_with_pause(f"{i}.", comment['body'], voice, comment_audio_path):
                logger.error(f"Failed to generate comment audio {i+1}")
                processing_queue[process_id]['status'] = 'failed'
                processing_queue[process_id]['message'] = f'Failed to generate comment audio {i+1}'
                return
            comment_audio_paths.append(comment_audio_path)
        
        # Step 3: Process audio with pauses (80%)
        processing_queue[process_id]['progress'] = 80
        processing_queue[process_id]['message'] = 'Processing audio with pauses...'
        
        # Combine comments with 1-second silences and track start times
        final_comment_audio, comment_timing_info = combine_comments_with_silences(comment_audio_paths, process_id)
        logger.info(f"Comments combined with {len(comment_audio_paths)-1} x 1-second pauses between {len(comment_audio_paths)} comments")
        
        # Combine template + comment audio
        final_audio_path = f"final_comment_audio_{process_id}.wav"
        combine_success, timing_info = combine_template_and_comments(
            template_audio_path, final_comment_audio, final_audio_path, comment_timing_info
        )
        
        if not combine_success:
            logger.error("Failed to combine audio files")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = 'Failed to combine audio files'
            return
        
        # Get media settings
        gameplay_video = user_options.get('gameplay', settings['defaultGameplay']) if user_options else settings['defaultGameplay']
        background_music = user_options.get('music', settings['defaultMusic']) if user_options else settings['defaultMusic']
        
        # Generate Reddit post image for template
        username = user_options.get('username', settings['defaultUsername']) if user_options else settings['defaultUsername']
        avatar_path = user_options.get('avatar', settings['defaultAvatar']) if user_options else settings['defaultAvatar']
        
        reddit_image = generate_reddit_post_image(title, username, avatar_path)
        
        # Generate video filename
        safe_title = create_safe_filename(title, max_words=5)
        final_video_path = f"output/{safe_title}_comments.mp4"
        
        # Step 4: Generate subtitles (85%)
        processing_queue[process_id]['progress'] = 85
        processing_queue[process_id]['message'] = 'Generating subtitles...'
        
        subtitle_path = None
        if whisper_model and final_comment_audio:
            logger.info("Generating Whisper subtitles for comments only (no template subtitles)...")
            # Generate subtitles from comments audio only and offset by template + silence duration
            if timing_info:
                # Account for template duration + 1 second silence between template and comments
                subtitle_start_offset = timing_info['template_duration'] + timing_info['silence_duration']
                logger.info(f"Subtitle offset: template ({timing_info['template_duration']:.2f}s) + silence ({timing_info['silence_duration']:.2f}s) = {subtitle_start_offset:.2f}s")
            else:
                subtitle_start_offset = 11.0  # Fallback: 10s template + 1s silence
            
            comment_start_times = timing_info.get('comment_start_times', []) if timing_info else []
            subtitle_segments = generate_whisper_subtitles(final_comment_audio, start_offset=subtitle_start_offset, comment_start_times=comment_start_times)
            
            if subtitle_segments:
                subtitle_path = f"subtitles_{process_id}.ass"
                if create_ass_subtitle_file(subtitle_segments, subtitle_path):
                    logger.info(f"Subtitles generated: {subtitle_path}")
                else:
                    logger.warning("Failed to create subtitle file")
                    subtitle_path = None
            else:
                logger.warning("No subtitle segments generated")
        else:
            if not whisper_model:
                logger.info("Skipping subtitle generation (Whisper not available)")
            else:
                logger.info("Skipping subtitle generation (no audio path)")
        
        # Step 5: Generate video (90%)
        processing_queue[process_id]['progress'] = 90
        processing_queue[process_id]['message'] = 'Generating final video...'
        
        # Generate final video with comment compilation timing
        if not generate_comment_compilation_video(reddit_image, final_audio_path, gameplay_video, background_music, final_video_path, timing_info, subtitle_path):
            logger.error("Failed to generate final video")
            processing_queue[process_id]['status'] = 'failed'
            processing_queue[process_id]['message'] = 'Failed to generate final video'
            return
        
        # Cleanup
        cleanup_files = [template_audio_path, final_comment_audio, final_audio_path] + comment_audio_paths
        if subtitle_path:
            cleanup_files.append(subtitle_path)
        if reddit_image and os.path.exists(reddit_image):
            cleanup_files.append(reddit_image)
        safe_cleanup_files(cleanup_files, "comment compilation temporary files")
        
        # Complete
        processing_queue[process_id]['progress'] = 100
        processing_queue[process_id]['message'] = 'Comment compilation complete!'
        processing_queue[process_id]['status'] = 'completed'
        processing_queue[process_id]['results'] = {
            'message': 'Comment compilation generated successfully!',
            'files': [os.path.basename(final_video_path)]
        }
        
        logger.info(f"Comment compilation completed successfully: {process_id}")
        
    except Exception as e:
        logger.error(f"Comment compilation failed for {process_id}: {str(e)}")
        processing_queue[process_id]['status'] = 'failed'
        processing_queue[process_id]['message'] = f'Comment compilation failed: {str(e)}'

def combine_comments_with_silences(comment_audio_paths, process_id):
    """Combine comment audio files with 1-second silences between them and track start times"""
    try:
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.error("Audio processing not available")
            return None, None
        
        # Load all comment audio segments
        comment_segments = []
        for audio_path in comment_audio_paths:
            segment = AudioSegment.from_wav(audio_path)
            comment_segments.append(segment)
        
        # Create 1-second silence
        silence_segment = AudioSegment.silent(duration=1000)  # 1 second
        
        # Combine segments with silences and track start times
        combined = AudioSegment.empty()
        comment_start_times = []  # Track when each comment starts
        current_time = 0.0
        total_comment_duration = 0
        total_pause_duration = 0
        
        for i, segment in enumerate(comment_segments):
            # Record when this comment starts within the combined audio
            comment_start_times.append(current_time)
            logger.info(f"Comment {i+1} starts at {current_time:.2f}s in combined audio")
            
            # Add the comment audio
            combined += segment
            comment_duration = len(segment) / 1000.0
            total_comment_duration += comment_duration
            current_time += comment_duration
            
            # Add silence after each comment except the last one
            if i < len(comment_segments) - 1:
                combined += silence_segment
                total_pause_duration += 1.0  # 1 second pause
                current_time += 1.0  # Move timing forward by silence duration
        
        logger.info(f"Combined comments: {total_comment_duration:.2f}s speech + {total_pause_duration:.2f}s pauses = {len(combined)/1000.0:.2f}s total")
        logger.info(f"Comment start times (relative to combined audio): {[f'{t:.2f}s' for t in comment_start_times]}")
        
        # Export combined audio
        output_path = f"combined_comments_{process_id}.wav"
        combined.export(output_path, format="wav")
        
        # Return both audio path and timing info
        timing_info = {
            'comment_start_times': comment_start_times,
            'total_duration': len(combined) / 1000.0
        }
        
        return output_path, timing_info
        
    except Exception as e:
        logger.error(f"Error combining comments with silences: {e}")
        return None, None

def generate_comment_compilation_video(reddit_image, audio_path, gameplay_video=None, background_music=None, output_path="output.mp4", timing_info=None, subtitle_path=None):
    """Generate video for comment compilation with proper timing"""
    try:
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.error("Video processing not available")
            return False
        
        # Detect hardware acceleration early for consistent use
        hw_encoder = detect_hardware_acceleration()
        logger.info(f"Generating comment compilation video at 60 fps with {hw_encoder} acceleration: {output_path}")
        
        # Get audio duration
        audio_duration = timing_info['total_duration'] if timing_info else 60.0
        template_duration = timing_info['template_duration'] if timing_info else 10.0
        
        if timing_info:
            logger.info(f"Comment compilation timing - Template: {timing_info['template_duration']:.2f}s, "
                       f"Silence: {timing_info['silence_duration']:.2f}s, "
                       f"Comments: {timing_info['comment_duration']:.2f}s, "
                       f"Total: {timing_info['total_duration']:.2f}s")
        
        # Set up video input (background video)
        if gameplay_video and os.path.exists(f"gameplay_videos/{gameplay_video}"):
            video_path = f"gameplay_videos/{gameplay_video}"
            logger.info(f"Using background video: {video_path}")
        else:
            logger.warning("No background video found, using black background")
            video_path = None
        
        # Create video stream
        if video_path:
            # Use background video with proper looping
            video_input = ffmpeg.input(video_path)
            video_stream = ffmpeg.filter(video_input, 'scale', 1080, 1920)  # TikTok aspect ratio
            # Apply hardware-accelerated video processing optimization
            video_probe = ffmpeg.probe(video_path)
            video_stream = optimize_video_processing(video_input, video_probe, hw_encoder)
            video_stream = ffmpeg.filter(video_stream, 'trim', duration=audio_duration)
            video_stream = ffmpeg.filter(video_stream, 'setpts', 'PTS-STARTPTS')  # Reset timestamps
        else:
            # Black background fallback at 60 fps
            video_stream = ffmpeg.input('color=black:size=1080x1920:rate=60', f='lavfi', t=audio_duration)
        
        # Add template overlay during title reading
        if reddit_image and os.path.exists(reddit_image):
            logger.info(f"Adding template overlay for first {template_duration:.2f}s")
            
            # Load template image and position it directly on the background video
            template_input = ffmpeg.input(reddit_image)
            
            # Get original template dimensions to calculate scaled height
            try:
                temp_img = Image.open(reddit_image)
                original_width, original_height = temp_img.size
                temp_img.close()
            except:
                # Fallback dimensions if image can't be read
                original_width, original_height = 448, 300
            
            # Scale template to fit nicely without black borders - make it fill most of the width
            template_width = int(1080 * 0.8)  # 80% of video width
            scale_factor = template_width / original_width
            template_height = int(original_height * scale_factor)  # Calculate scaled height
            
            scaled_template = ffmpeg.filter(template_input, 'scale', template_width, template_height)
            
            # Position template in center of video (centered horizontally and vertically)
            template_x = int((1080 - template_width) / 2)  # Center horizontally with proper rounding
            template_y = int((1920 - template_height) / 2)  # Center vertically with proper rounding
            
            # Overlay template during title reading only - positioned on top of background video
            video_stream = ffmpeg.filter(
                [video_stream, scaled_template], 
                'overlay', 
                x=template_x, y=template_y,
                enable=f'between(t,0,{template_duration})'
            )
            
            logger.info(f"Template CENTERED at ({template_x}, {template_y}) with size {template_width}x{template_height}px")
            logger.info(f"[DEBUG] Centering calculation: (1080 - {template_width}) / 2 = {template_x}, (1920 - {template_height}) / 2 = {template_y}")
        else:
            logger.warning("No template image found, skipping overlay")
        
        # Audio processing
        audio_streams = []
        
        # Main audio (title + comments)
        main_audio = ffmpeg.input(audio_path)
        audio_streams.append(main_audio)
        
        # Background music
        if background_music and os.path.exists(f"background_music/{background_music}"):
            music_path = f"background_music/{background_music}"
            logger.info(f"Adding background music: {music_path}")
            
            try:
                music_input = ffmpeg.input(music_path)
                music_stream = ffmpeg.filter(music_input, 'aloop', loop=-1, size=2e+09)  # Loop audio
                music_stream = ffmpeg.filter(music_stream, 'volume', 0.35)  # 25% volume
                music_stream = ffmpeg.filter(music_stream, 'atrim', duration=audio_duration)
                audio_streams.append(music_stream)
            except Exception as e:
                logger.warning(f"Could not process background music: {e}")
        
        # Mix audio streams
        if len(audio_streams) > 1:
            mixed_audio = ffmpeg.filter(audio_streams, 'amix', inputs=len(audio_streams), duration='longest')
        else:
            mixed_audio = audio_streams[0]
        
        # Apply subtitles if available - only during comments portion (after template)
        if subtitle_path and os.path.exists(subtitle_path):
            logger.info(f"Adding subtitles from: {subtitle_path} (comments only - no template subtitles)")
            
            # Convert Windows paths to forward slashes for ffmpeg
            subtitle_path_ffmpeg = subtitle_path.replace('\\', '/')
            video_stream = ffmpeg.filter(video_stream, 'subtitles', subtitle_path_ffmpeg)
        else:
            logger.info("No subtitles to add")
        
        # Build output args with hardware acceleration and 60 fps
        output_args = build_output_args(hw_encoder, audio_duration)
        output_args['r'] = 60  # Override frame rate to 60 fps
        output_args['movflags'] = 'faststart'
        output_args['pix_fmt'] = 'yuv420p'
        
        # Output with hardware acceleration and 60 fps for smoother playback
        output = ffmpeg.output(
            video_stream, mixed_audio,
            output_path,
            **output_args
        )
        
        # Run ffmpeg with hardware acceleration and better error handling
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Starting 60 fps hardware-accelerated video encoding ({hw_encoder}) for comment compilation...")
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            encoding_time = time.time() - start_time
            logger.info(f"Comment compilation encoding completed in {encoding_time:.2f} seconds with {hw_encoder}")
            
        except ffmpeg.Error as e:
            logger.error(f"ffmpeg encoding failed:")
            if e.stdout:
                logger.error(f"stdout: {e.stdout.decode('utf-8', errors='replace')}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr.decode('utf-8', errors='replace')}")
            return False
        except Exception as e:
            logger.error(f"Unexpected encoding error: {e}")
            return False
        
        logger.info(f"Comment compilation video generated successfully: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating comment compilation video: {e}")
        return False

def process_comment_audio_with_pauses(comment_audio_path, process_id):
    """Process comment audio to replace [PAUSE] markers with 1-second silences"""
    try:
        # Load the audio
        audio = AudioSegment.from_wav(comment_audio_path)
        
        # For now, return the original audio - we'll implement pause processing later
        # This is a placeholder that can be enhanced to detect [PAUSE] markers
        output_path = f"processed_comment_audio_{process_id}.wav"
        audio.export(output_path, format="wav")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing comment audio with pauses: {e}")
        return comment_audio_path

def combine_template_and_comments(template_audio_path, comment_audio_path, output_path, comment_timing_info=None):
    """Combine template and comment audio files with comment timing information"""
    try:
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.error("Audio processing not available")
            return False, None
        
        # Load audio segments
        template_segment = AudioSegment.from_wav(template_audio_path)
        comment_segment = AudioSegment.from_wav(comment_audio_path)
        
        # Add 1 second silence between template and comments
        silence_segment = AudioSegment.silent(duration=1000)
        
        # Combine: template + silence + comments
        combined = template_segment + silence_segment + comment_segment
        
        # Export combined audio
        combined.export(output_path, format="wav")
        
        # Create timing info with comment start times
        template_duration = len(template_segment) / 1000.0
        silence_duration = 1.0
        comment_start_offset = template_duration + silence_duration  # When comments start in final audio
        
        timing_info = {
            'template_duration': template_duration,
            'silence_duration': silence_duration,
            'comment_duration': len(comment_segment) / 1000.0,
            'total_duration': len(combined) / 1000.0,
            'comment_start_offset': comment_start_offset
        }
        
        # Add individual comment start times (adjusted for template + silence)
        if comment_timing_info and 'comment_start_times' in comment_timing_info:
            # Adjust comment start times to be relative to final combined audio
            adjusted_start_times = [
                comment_start_offset + start_time 
                for start_time in comment_timing_info['comment_start_times']
            ]
            timing_info['comment_start_times'] = adjusted_start_times
            logger.info(f"Final comment start times in complete audio: {[f'{t:.2f}s' for t in adjusted_start_times]}")
        else:
            logger.warning("No comment timing information provided")
        
        return True, timing_info
        
    except Exception as e:
        logger.error(f"Error combining template and comment audio: {e}")
        return False, None

def validate_reddit_url(url):
    """Validate if the provided URL is a valid Reddit post URL"""
    try:
        parsed = urlparse(url)
        if 'reddit.com' not in parsed.netloc:
            return False
        # Check if it's a post URL pattern
        if '/comments/' not in parsed.path:
            return False
        return True
    except Exception:
        return False

def extract_post_id_from_url(url):
    """Extract post ID from Reddit URL"""
    try:
        # Pattern: /r/subreddit/comments/post_id/title
        match = re.search(r'/comments/([a-zA-Z0-9]+)/', url)
        if match:
            return match.group(1)
        return None
    except Exception:
        return None

def save_post_to_file(title, author, content, filename='scraped_post.txt'):
    """Save scraped post data to a text file with generated TikTok hook and single-paragraph content"""
    try:
        # Generate TikTok hook
        hook = generate_tiktok_hook(title, content)
        
        # Clean up content
        # 1. Remove markdown links
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        # 2. Replace all types of line breaks with spaces
        cleaned_content = content.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
        # 3. Replace multiple spaces with single space
        cleaned_content = ' '.join(cleaned_content.split())
        
        # Clean up title (ensure it's on one line)
        title = ' '.join(title.split())
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Title: {title}\n")
            f.write(f"TikTok Hook: {hook}\n")
            f.write(f"Author: {author}\n")
            f.write(f"Content: {cleaned_content}")  # No newline after Content:
        return True
    except Exception as e:
        logger.error(f"Error saving post to file: {e}")
        return False

def save_urls_to_file(urls, subreddit_name, filename=None):
    """Save top post URLs to a text file"""
    try:
        if not filename:
            filename = f"top_posts_{subreddit_name}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for i, url in enumerate(urls, 1):
                f.write(f"{i}. {url}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving URLs to file: {e}")
        return False

def save_individual_posts(posts, subreddit_name):
    """Save individual post content to separate files and return list of filenames"""
    saved_files = []
    
    for i, post in enumerate(posts, 1):
        try:
            # Get post data
            title = post.title
            author = f"u/{post.author.name}" if post.author else "[deleted]"
            content = post.selftext if post.selftext else "[No text content]"
            
            # Create safe filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title[:50]  # Limit length
            filename = f"{subreddit_name}_post_{i:02d}_{safe_title}.txt"
            
            # Generate TikTok hook
            hook = generate_tiktok_hook(title, content)
            
            # Clean up content
            # 1. Remove markdown links
            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
            # 2. Replace all types of line breaks with spaces
            cleaned_content = content.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
            # 3. Replace multiple spaces with single space
            cleaned_content = ' '.join(cleaned_content.split())
            
            # Clean up title (ensure it's on one line)
            title = ' '.join(title.split())
            
            # Save individual post
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n")
                f.write(f"TikTok Hook: {hook}\n")
                f.write(f"Author: {author}\n")
                f.write(f"URL: https://www.reddit.com{post.permalink}\n")
                f.write(f"Content: {cleaned_content}")
            
            saved_files.append(filename)
            logger.debug(f"[DEBUG] Saved individual post: {filename}")
            
        except Exception as e:
            logger.debug(f"[DEBUG] Failed to save post {i}: {e}")
            continue
    
    return saved_files

def cleanup_generated_files():
    # Delete all .txt files except prd.md, requirements.txt, app.py
    keep_files = {"prd.md", "requirements.txt", "app.py", "settings.json"}
    for file in glob.glob("*.txt"):
        if file not in keep_files:
            try:
                os.remove(file)
                logger.debug(f"[DEBUG] Deleted file: {file}")
            except Exception as e:
                logger.debug(f"[DEBUG] Failed to delete {file}: {e}")

def safe_cleanup_files(file_paths, description="temporary files"):
    """Safely delete files with error handling"""
    if not file_paths:
        return
    
    cleaned_count = 0
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                cleaned_count += 1
        except Exception as e:
            logger.warning(f"[WARNING] Could not delete {file_path}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} {description}")

def convert_comment_numbers_to_digits(text, segment_start_time=0.0, comment_start_times=None):
    """Convert written numbers 1-10 to digits, but only during first 2 seconds of each comment
    
    This function applies number conversion only during the first 2 seconds of each
    individual comment to catch comment numbering (1., 2., 3.) while preserving 
    natural speech numbers that occur later in each comment.
    
    Args:
        text (str): The subtitle text to potentially convert
        segment_start_time (float): Start time of this subtitle segment in seconds
        comment_start_times (list): List of start times for each comment
        
    Returns:
        str: Text with comment numbering converted to digits (if applicable)
    """
    import re
    
    original_text = text  # Store original for logging
    
    # Only apply conversion logic when we're at the very start of individual comments
    # Strategy: Apply conversion only if we're within the first 2 seconds of any comment start
    
    COMMENT_START_WINDOW = 2.0  # Only convert numbers in first 2 seconds of each comment
    
    # Quick timing check first - if we're not in any comment start window, skip everything
    is_in_comment_start = False
    closest_comment_start = None
    
    if comment_start_times:
        for comment_start in comment_start_times:
            # Check if this subtitle segment starts within 2 seconds of any comment start
            time_from_comment_start = segment_start_time - comment_start
            if 0 <= time_from_comment_start <= COMMENT_START_WINDOW:
                is_in_comment_start = True
                closest_comment_start = comment_start
                break
    
    # Early return if not in comment start window - no need to check words or log
    if not is_in_comment_start:
        return text  # Skip all processing - we're not at a comment start
    
    # Now we know we're in a comment start window - check for number words
    has_number_at_start = False
    for word in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']:
        if re.match(rf'^{word}\b', text.lower()):
            has_number_at_start = True
            break
    
    # Log what we're processing (only for segments in comment start windows)
    if has_number_at_start:
        time_from_start = segment_start_time - closest_comment_start
        logger.info(f"[DEBUG] Processing number text at {segment_start_time:.2f}s (+{time_from_start:.2f}s from comment start): '{text}'")
    
    if not has_number_at_start:
        return text  # No number word at start, nothing to convert
    
    # Dictionary mapping written numbers to digits (1-10 only)
    number_words = {
        'one': '1',
        'two': '2', 
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10'
    }
    
    # Convert comment numbering with simple natural speech protection
    for word_num, digit_num in number_words.items():
        # Check if text starts with this number word
        if re.match(rf'^{word_num}\b', text, re.IGNORECASE):
            logger.info(f"[DEBUG] Found number word '{word_num}' at start of: '{text}'")
            
            # Simple check: skip only the most obvious natural speech patterns
            skip_patterns = [
                rf'^{word_num}\s+of\b',     # "one of"
                rf'^{word_num}\s+the\b',    # "two the" (rare but possible)
                rf'^{word_num}\s+day\b',    # "one day"
            ]
            
            should_skip = False
            for skip_pattern in skip_patterns:
                if re.match(skip_pattern, text, re.IGNORECASE):
                    logger.info(f"[DEBUG] Skipping natural speech: '{text}'")
                    should_skip = True
                    break
            
            if should_skip:
                continue
                
            # Convert to comment numbering - handle all formats and remove unwanted punctuation
            logger.info(f"[DEBUG] Converting '{word_num}' to '{digit_num}.' in: '{text}'")
            
            if re.match(rf'^{word_num}\.', text, re.IGNORECASE):
                # "one." -> "1."
                text = re.sub(rf'^{word_num}\.', f'{digit_num}.', text, flags=re.IGNORECASE)
            elif re.match(rf'^{word_num},', text, re.IGNORECASE):
                # "one," -> "1." (remove unwanted comma)
                logger.info(f"[DEBUG] Removing comma from: '{text}'")
                text = re.sub(rf'^{word_num},\s*', f'{digit_num}. ', text, flags=re.IGNORECASE)
            elif re.match(rf'^{word_num}\s', text, re.IGNORECASE):
                # "one " -> "1. "
                text = re.sub(rf'^{word_num}\s+', f'{digit_num}. ', text, flags=re.IGNORECASE)
            elif re.match(rf'^{word_num}$', text, re.IGNORECASE):
                # "one" -> "1."
                text = f'{digit_num}.'
            else:
                # Fallback - handle any punctuation after the number
                text = re.sub(rf'^{word_num}[^\w\s]*\s*', f'{digit_num}. ', text, flags=re.IGNORECASE)
            
            logger.info(f"[DEBUG] Result after conversion: '{text}'")
            break  # Only convert the first matching number
    
    # Log if conversion happened (or didn't happen for natural speech)
    if text != original_text:
        if closest_comment_start is not None:
            time_from_start = segment_start_time - closest_comment_start
            logger.info(f"Comment numbering converted at {segment_start_time:.2f}s (+{time_from_start:.2f}s from comment start): '{original_text}' -> '{text}'")
        else:
            logger.info(f"Comment numbering converted at {segment_start_time:.2f}s: '{original_text}' -> '{text}'")
    elif any(word in original_text.lower().split()[:2] for word in number_words.keys()):
        logger.info(f"Natural speech preserved at {segment_start_time:.2f}s: '{original_text}'")
    
    return text

def generate_whisper_subtitles(audio_path, start_offset=0.0, comment_start_times=None):
    """
    Generate subtitles using Whisper with word-level timestamps
    
    Args:
        audio_path (str): Path to audio file for transcription
        start_offset (float): Time offset in seconds to add to all timestamps
        comment_start_times (list): List of comment start times for numbering logic
        
    Returns:
        list: List of subtitle segments with word-level timing
    """
    try:
        if not whisper_model:
            logger.error("Whisper model not available")
            return []
            
        logger.info(f"Generating Whisper subtitles for: {audio_path}")
        
        # Transcribe with word-level timestamps
        result = whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False
        )
        
        subtitle_segments = []
        
        # Process each segment
        for segment in result["segments"]:
            if "words" in segment:
                # Group words based on timing (≤ 0.375 seconds for 2 words)
                grouped_words = group_words_by_timing(segment["words"])
                
                for word_group in grouped_words:
                    # Calculate timing with offset
                    start_time = word_group[0]["start"] + start_offset
                    end_time = word_group[-1]["end"] + start_offset
                    
                    # Combine words in group
                    text = " ".join([word["word"].strip() for word in word_group])
                    
                    # Convert comment numbering 1-10 only at the start of comments (first 2 seconds of each comment)
                    # "one. This is my story" -> "1. This is my story" (if within comment start window)
                    # "one," -> "1. " (removes comma, if within comment start window) 
                    # "one of the best movies" -> "one of the best movies" (always preserved)
                    text = convert_comment_numbers_to_digits(text.strip(), start_time, comment_start_times)
                    
                    subtitle_segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    })
        
        logger.info(f"Generated {len(subtitle_segments)} subtitle segments")
        return subtitle_segments
        
    except Exception as e:
        logger.error(f"Failed to generate Whisper subtitles: {e}")
        return []

def group_words_by_timing(words, max_duration=0.375):
    """
    Group words based on timing constraints
    
    Args:
        words (list): List of word dictionaries with start/end times
        max_duration (float): Maximum duration for word grouping (default: 0.375 seconds)
        
    Returns:
        list: List of word groups
    """
    if not words:
        return []
    
    groups = []
    current_group = [words[0]]
    
    for i in range(1, len(words)):
        current_word = words[i]
        group_start = current_group[0]["start"]
        group_end = current_word["end"]
        
        # Check if adding this word would exceed max duration
        if (group_end - group_start) <= max_duration and len(current_group) < 2:
            current_group.append(current_word)
        else:
            # Start new group
            groups.append(current_group)
            current_group = [current_word]
    
    # Add the last group
    if current_group:
        groups.append(current_group)
    
    return groups

def create_ass_subtitle_file(subtitle_segments, output_path):
    """
    Create ASS subtitle file for 1080x1920 video format
    
    Args:
        subtitle_segments (list): List of subtitle segments
        output_path (str): Path to save ASS file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Creating ASS subtitle file: {output_path}")
        
        # Use system font name for reliable recognition by libass
        selected_font = "Bangers"
        
        # ASS file header for 1080x1920 format with Bangers font
        ass_header = f"""[Script Info]
Title: TikTok Video Subtitles
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{selected_font},200,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,7,0,5,50,50,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ass_header)
            
            # Write subtitle events
            for segment in subtitle_segments:
                start_time = format_ass_time(segment["start"])
                end_time = format_ass_time(segment["end"])
                text = segment["text"]
                
                # ASS dialogue line
                dialogue_line = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"
                f.write(dialogue_line)
        
        logger.info(f"ASS subtitle file created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create ASS subtitle file: {e}")
        return False

def format_ass_time(seconds):
    """
    Format time in seconds to ASS time format (H:MM:SS.CC)
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def calculate_subtitle_timing_offset(hook_duration, story_by_duration):
    """
    Calculate the timing offset for when subtitles should start
    
    Args:
        hook_duration (float): Duration of hook audio in seconds
        story_by_duration (float): Duration of "story by" audio in seconds
        
    Returns:
        float: Time offset in seconds for subtitle start
    """
    # Subtitles start when "Story by:" begins
    # This is after the hook but includes the "Story by:" part
    return hook_duration

def generate_story_by_text_overlay(story_by_text, output_path):
    """
    Generate a text overlay image for "Story by: username" text
    Matches ASS subtitle styling (Arial, 72pt, white with black outline)
    
    Args:
        story_by_text (str): The story by text to display
        output_path (str): Path to save the overlay image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.error("[ERROR] Video processing not available - cannot generate text overlay")
            return False
            
        logger.info(f"[INFO] Generating story by text overlay: {story_by_text}")
        
        # Create transparent image (1080x1920 to match video format)
        width = 1080
        height = 1920
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Load font to match ASS subtitle style (Bangers-Regular, 72pt, bold)
        font_size = 72
        try:
            # Try to load Bangers-Regular font
            font = ImageFont.truetype("Bangers-Regular.ttf", font_size)
        except (OSError, IOError):
            try:
                # Fallback to system default font
                font = ImageFont.load_default()
                logger.warning("[WARNING] Bangers-Regular font not found, using default font")
            except:
                logger.error("[ERROR] Could not load any font")
                return False
        
        # Calculate text dimensions for centering
        bbox = draw.textbbox((0, 0), story_by_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position text at center of screen (matching ASS alignment 5)
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text with black outline (4 pixels) to match ASS style
        outline_width = 4
        
        # Draw outline by drawing text in black in 8 directions
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), story_by_text, font=font, fill=(0, 0, 0, 255))
        
        # Draw main text in white
        draw.text((x, y), story_by_text, font=font, fill=(255, 255, 255, 255))
        
        # Save the overlay image
        image.save(output_path)
        logger.info(f"[INFO] Story by text overlay saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate story by text overlay: {e}")
        return False

# New API endpoints for the enhanced features

@app.route('/upload_media', methods=['POST'])
def upload_media():
    """Upload media files (gameplay videos, background music, avatar images)"""
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        file_type = request.form.get('type')
        
        if file_type not in ['gameplay', 'music', 'avatar']:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        ensure_folders()
        
        uploaded_files = []
        folder_mapping = {
            'gameplay': 'gameplay_videos',
            'music': 'background_music',
            'avatar': 'avatar_images'
        }
        
        upload_folder = folder_mapping[file_type]
        
        for file in files:
            if file and file.filename and allowed_file(file.filename, file_type):
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_folder, filename)
                
                # Check if file already exists
                counter = 1
                base_name, ext = os.path.splitext(filename)
                while os.path.exists(filepath):
                    filename = f"{base_name}_{counter}{ext}"
                    filepath = os.path.join(upload_folder, filename)
                    counter += 1
                
                file.save(filepath)
                uploaded_files.append(filename)
        
        return jsonify({'success': True, 'files': uploaded_files})
        
    except Exception as e:
        logger.error(f"Error uploading media: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_media/<file_type>')
def get_media(file_type):
    """Get list of media files"""
    try:
        if file_type not in ['gameplay', 'music', 'avatar']:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        folder_mapping = {
            'gameplay': 'gameplay_videos',
            'music': 'background_music',
            'avatar': 'avatar_images'
        }
        
        folder = folder_mapping[file_type]
        ensure_folders()
        
        files = []
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    file_info = get_file_info(filepath)
                    if file_info:
                        files.append(file_info)
        
        return jsonify({'success': True, 'files': files})
        
    except Exception as e:
        logger.error(f"Error getting media: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/media/<file_type>/<filename>')
def serve_media(file_type, filename):
    """Serve media files"""
    try:
        folder_mapping = {
            'gameplay': 'gameplay_videos',
            'music': 'background_music',
            'avatar': 'avatar_images'
        }
        
        if file_type not in folder_mapping:
            return "Invalid file type", 404
        
        folder = folder_mapping[file_type]
        return send_from_directory(folder, filename)
        
    except Exception as e:
        logger.error(f"Error serving media: {e}")
        return f"File not found: {str(e)}", 404

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """Save application settings"""
    try:
        global settings
        new_settings = request.get_json()
        settings.update(new_settings)
        
        if save_settings_to_file():
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to save settings'}), 500
            
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_settings')
def get_settings():
    """Get application settings"""
    try:
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/scrape_and_generate', methods=['POST'])
def scrape_and_generate():
    """Scrape Reddit post and generate video"""
    try:
        url = request.form.get('url')
        mode = request.form.get('mode', 'auto')
        
        if not url or not validate_reddit_url(url):
            return jsonify({'success': False, 'error': 'Invalid Reddit URL'}), 400
        
        # Extract post ID and get post data
        post_id = extract_post_id_from_url(url)
        if not post_id:
            return jsonify({'success': False, 'error': 'Could not extract post ID'}), 400
        
        # Scrape Reddit post
        logger.info(f"[DEBUG] Fetching Reddit post with ID: {post_id}")
        submission = reddit.submission(id=post_id)
        if submission.author is None:
            return jsonify({'success': False, 'error': 'Post not found or deleted'}), 404
        
        title = submission.title
        author = f"u/{submission.author.name}" if submission.author else "[deleted]"
        content = submission.selftext if submission.selftext else "[No text content]"
        
        logger.info(f"[DEBUG] Extracted post data:")
        logger.info(f"[DEBUG] Title: {title[:100]}...")
        logger.info(f"[DEBUG] Author: {author}")
        logger.info(f"[DEBUG] Content length: {len(content)} characters")
        logger.info(f"[DEBUG] Content preview: {content[:200]}...")
        
        hook = generate_tiktok_hook(title, content)
        logger.info(f"[DEBUG] Generated hook: {hook}")
        
        post_data = {
            'title': title,
            'author': author,
            'content': content,
            'hook': hook
        }
        
        # Get user options if in user mode
        user_options = None
        if mode == 'user':
            custom_username = request.form.get('customUsername', settings['defaultUsername'])
            logger.info(f"[DEBUG] Custom username from form: '{custom_username}'")
            user_options = {
                'voice': request.form.get('ttsVoice', 'p232'),
                'username': custom_username,
                'gameplay': request.form.get('gameplaySelect', settings['defaultGameplay']),
                'music': request.form.get('musicSelect', settings['defaultMusic']),
                'avatar': request.form.get('avatarSelect', settings['defaultAvatar'])
            }
            logger.info(f"[DEBUG] User options created: {user_options}")
            
            # Validate user options for problematic values
            for key, value in user_options.items():
                if value is not None and str(value).strip() == '':
                    logger.warning(f"[WARNING] Empty user option detected: {key} = '{value}', this may cause issues")
        
        # Start video generation process
        process_id = str(uuid.uuid4())
        thread = threading.Thread(
            target=process_video_generation,
            args=(process_id, post_data, mode, user_options)
        )
        thread.start()
        
        return jsonify({'success': True, 'processId': process_id})
        
    except Exception as e:
        logger.error(f"Error scraping and generating: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/scrape_and_generate_bulk', methods=['POST'])
def scrape_and_generate_bulk():
    """Scrape subreddit and generate videos for all posts"""
    try:
        subreddit_name = request.form.get('subreddit')
        limit = int(request.form.get('limit', 10))
        sort_type = request.form.get('sort', 'hot')
        mode = request.form.get('mode', 'auto')
        
        if not subreddit_name:
            return jsonify({'success': False, 'error': 'Subreddit name required'}), 400
        
        # Clean subreddit name
        subreddit_name = subreddit_name.replace('r/', '').strip()
        
        # Get subreddit posts
        subreddit = reddit.subreddit(subreddit_name)
        
        # Fetch posts based on sort type
        if sort_type == 'hot':
            posts = list(subreddit.hot(limit=limit * 2))
        elif sort_type == 'top_all':
            posts = list(subreddit.top(time_filter='all', limit=limit * 2))
        elif sort_type == 'top_week':
            posts = list(subreddit.top(time_filter='week', limit=limit * 2))
        elif sort_type == 'top_month':
            posts = list(subreddit.top(time_filter='month', limit=limit * 2))
        else:
            posts = list(subreddit.hot(limit=limit * 2))
        
        # Filter out stickied posts
        filtered_posts = [post for post in posts if not post.stickied][:limit]
        
        if not filtered_posts:
            return jsonify({'success': False, 'error': 'No posts found'}), 404
        
        # Get user options if in user mode
        user_options = None
        if mode == 'user':
            user_options = {
                'voice': request.form.get('ttsVoice', 'p232'),
                'username': request.form.get('customUsername', settings['defaultUsername']),
                'gameplay': request.form.get('gameplaySelect', settings['defaultGameplay']),
                'music': request.form.get('musicSelect', settings['defaultMusic']),
                'avatar': request.form.get('avatarSelect', settings['defaultAvatar'])
            }
        
        # Start bulk processing
        process_id = str(uuid.uuid4())
        
        # Initialize queue with all posts
        processing_queue[process_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Processing multiple posts...',
            'details': f'0 of {len(filtered_posts)} completed',
            'queue': [],
            'total_posts': len(filtered_posts),
            'completed_posts': 0
        }
        
        # Add posts to queue
        for i, post in enumerate(filtered_posts):
            processing_queue[process_id]['queue'].append({
                'name': f"Post {i+1}: {post.title[:50]}...",
                'status': 'pending'
            })
        
        # Implement actual bulk processing logic
        def process_bulk_videos():
            try:
                completed_videos = []
                failed_posts = []
                
                for i, post in enumerate(filtered_posts):
                    try:
                        # Update queue status
                        processing_queue[process_id]['queue'][i]['status'] = 'processing'
                        processing_queue[process_id]['progress'] = int((i / len(filtered_posts)) * 90)
                        processing_queue[process_id]['message'] = f'Processing post {i+1}/{len(filtered_posts)}: {post.title[:30]}...'
                        processing_queue[process_id]['details'] = f'{i} of {len(filtered_posts)} completed'
                        processing_queue[process_id]['completed_posts'] = i
                        
                        # Create post data
                        title = post.title
                        author = f"u/{post.author.name}" if post.author else "[deleted]"
                        content = post.selftext if post.selftext else "[No text content]"
                        hook = generate_tiktok_hook(title, content)
                        
                        post_data = {
                            'title': title,
                            'author': author,
                            'content': content,
                            'hook': hook
                        }
                        
                        # Create individual video
                        video_process_id = f"{process_id}_post_{i}"
                        video_thread = threading.Thread(
                            target=process_video_generation,
                            args=(video_process_id, post_data, mode, user_options)
                        )
                        video_thread.start()
                        video_thread.join()  # Wait for completion
                        
                        # Check if video was created successfully
                        if video_process_id in processing_queue and processing_queue[video_process_id]['status'] == 'completed':
                            video_files = processing_queue[video_process_id]['results']['files']
                            completed_videos.extend(video_files)
                            processing_queue[process_id]['queue'][i]['status'] = 'completed'
                            logger.info(f"Completed video {i+1}/{len(filtered_posts)}: {title[:30]}...")
                        else:
                            failed_posts.append(f"Post {i+1}: {title[:50]}...")
                            processing_queue[process_id]['queue'][i]['status'] = 'failed'
                            logger.error(f"Failed to create video for post {i+1}: {title[:30]}...")
                        
                        # Clean up individual processing queue entry
                        if video_process_id in processing_queue:
                            del processing_queue[video_process_id]
                            
                    except Exception as e:
                        failed_posts.append(f"Post {i+1}: {title[:50]}... (Error: {str(e)})")
                        processing_queue[process_id]['queue'][i]['status'] = 'failed'
                        logger.error(f"Exception processing post {i+1}: {e}")
                        continue
                
                # Create zip file if we have videos
                if completed_videos:
                    processing_queue[process_id]['progress'] = 95
                    processing_queue[process_id]['message'] = 'Creating zip file...'
                    
                    zip_filename = f"bulk_videos_{process_id}.zip"
                    zip_path = os.path.join('output', zip_filename)
                    
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for video_file in completed_videos:
                            if os.path.exists(video_file):
                                arcname = os.path.basename(video_file)
                                zipf.write(video_file, arcname)
                                logger.info(f"Added {video_file} to zip as {arcname}")
                    
                    # Update final status
                    processing_queue[process_id]['status'] = 'completed'
                    processing_queue[process_id]['progress'] = 100
                    processing_queue[process_id]['message'] = 'Bulk processing completed!'
                    processing_queue[process_id]['details'] = f'{len(completed_videos)} videos created, {len(failed_posts)} failed'
                    processing_queue[process_id]['completed_posts'] = len(completed_videos)
                    processing_queue[process_id]['results'] = {
                        'message': f'Successfully processed {len(completed_videos)} videos, {len(failed_posts)} failed',
                        'files': [zip_filename],
                        'completed_videos': len(completed_videos),
                        'failed_posts': failed_posts
                    }
                    
                    logger.info(f"Bulk processing completed: {len(completed_videos)} videos, {len(failed_posts)} failed")
                else:
                    # No videos created
                    processing_queue[process_id]['status'] = 'failed'
                    processing_queue[process_id]['progress'] = 100
                    processing_queue[process_id]['message'] = 'No videos could be created'
                    processing_queue[process_id]['results'] = {
                        'message': 'All posts failed to process',
                        'files': [],
                        'completed_videos': 0,
                        'failed_posts': failed_posts
                    }
                    logger.error("Bulk processing failed: No videos could be created")
                    
            except Exception as e:
                processing_queue[process_id]['status'] = 'failed'
                processing_queue[process_id]['message'] = f'Bulk processing failed: {str(e)}'
                logger.error(f"Bulk processing exception: {e}")
        
        thread = threading.Thread(target=process_bulk_videos)
        thread.start()
        
        return jsonify({'success': True, 'processId': process_id})
        
    except Exception as e:
        logger.error(f"Error scraping and generating bulk: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/scrape_and_generate_comments', methods=['POST'])
def scrape_and_generate_comments():
    """Scrape Reddit post comments and generate compilation video"""
    try:
        comment_url = request.form.get('commentUrl')
        mode = request.form.get('mode', 'auto')
        
        if not comment_url or not validate_reddit_url(comment_url):
            return jsonify({'success': False, 'error': 'Invalid Reddit URL'}), 400
        
        # Extract post ID and get post data
        post_id = extract_post_id_from_url(comment_url)
        if not post_id:
            return jsonify({'success': False, 'error': 'Could not extract post ID'}), 400
        
        # Scrape Reddit post and comments
        logger.info(f"[DEBUG] Fetching Reddit post with ID: {post_id}")
        submission = reddit.submission(id=post_id)
        
        if submission.author is None:
            return jsonify({'success': False, 'error': 'Post not found or deleted'}), 404
        
        title = submission.title
        author = f"u/{submission.author.name}" if submission.author else "[deleted]"
        
        # Get and process comments
        comments = get_top_comments(submission)
        if not comments:
            return jsonify({'success': False, 'error': 'No suitable comments found'}), 404
        
        logger.info(f"[DEBUG] Found {len(comments)} suitable comments")
        
        # Create post data for comment compilation
        post_data = {
            'title': title,
            'author': author,
            'content': '',  # No content for comment compilation
            'hook': '',     # No hook for comment compilation
            'comments': comments,
            'compilation_type': 'comments'
        }
        
        # Get user options if in user mode
        user_options = None
        if mode == 'user':
            user_options = {
                'voice': request.form.get('ttsVoice', 'p232'),
                'username': request.form.get('customUsername', settings['defaultUsername']),
                'gameplay': request.form.get('gameplaySelect', settings['defaultGameplay']),
                'music': request.form.get('musicSelect', settings['defaultMusic']),
                'avatar': request.form.get('avatarSelect', settings['defaultAvatar'])
            }
        
        # Start video generation process
        process_id = str(uuid.uuid4())
        thread = threading.Thread(
            target=process_comment_compilation,
            args=(process_id, post_data, mode, user_options)
        )
        thread.start()
        
        return jsonify({'success': True, 'processId': process_id})
        
    except Exception as e:
        logger.error(f"Error scraping and generating comments: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_progress/<process_id>')
def get_progress(process_id):
    """Get progress of video generation process"""
    try:
        if process_id not in processing_queue:
            return jsonify({'success': False, 'error': 'Process not found'}), 404
        
        return jsonify(processing_queue[process_id])
        
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/cancel_process/<process_id>', methods=['POST'])
def cancel_process(process_id):
    """Cancel video generation process"""
    try:
        if process_id in processing_queue:
            processing_queue[process_id]['status'] = 'cancelled'
            processing_queue[process_id]['message'] = 'Process cancelled by user'
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Process not found'}), 404
            
    except Exception as e:
        logger.error(f"Error cancelling process: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up all generated files"""
    try:
        cleanup_generated_files()
        return jsonify({'success': True, 'message': 'Files cleaned up successfully'})
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def index():
    """Main page"""
    cleanup_generated_files()
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated files"""
    try:
        # Check in output folder first
        output_path = os.path.join('output', filename)
        if os.path.exists(output_path):
            return send_file(output_path, as_attachment=True)
        
        # Check in root directory
        if os.path.exists(filename):
            return send_file(filename, as_attachment=True)
        
        return "File not found", 404
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return f"Error downloading file: {str(e)}", 500

@app.route('/queue_status')
def get_queue_status():
    """Get current queue status and statistics"""
    try:
        active_jobs = [job for job in processing_queue.values() if job['status'] in ['processing', 'pending']]
        completed_jobs = [job for job in processing_queue.values() if job['status'] == 'completed']
        failed_jobs = [job for job in processing_queue.values() if job['status'] == 'failed']
        
        return jsonify({
            'success': True,
            'queue_stats': {
                'active': len(active_jobs),
                'completed': len(completed_jobs),
                'failed': len(failed_jobs),
                'total': len(processing_queue)
            }
        })
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/system_status')
def get_system_status():
    """Get comprehensive system status"""
    try:
        # Check model status
        models_status = {
            'llama': llm is not None,
            'tts': tts is not None,
            'whisper': whisper_model is not None,
            'video_processing': VIDEO_PROCESSING_AVAILABLE
        }
        
        # Get TTS speakers count
        speakers_count = len(tts.speakers) if tts else 0
        
        return jsonify({
            'success': True,
            'system': {
                'models_loaded': models_status,
                'tts_speakers': speakers_count,
                'queue_size': len(processing_queue)
            },
            'capabilities': {
                'reddit_scraping': True,
                'tts_generation': models_status['tts'],
                'hook_generation': models_status['llama'],
                'subtitle_generation': models_status['whisper'],
                'video_generation': models_status['video_processing'],
                'postfully_integration': True
            }
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def create_safe_filename(title, max_words=5):
    """
    Create a safe filename from the first few words of the title
    
    Args:
        title (str): The Reddit post title
        max_words (int): Maximum number of words to use (default: 5)
        
    Returns:
        str: Safe filename string
    """
    # Get first max_words words
    words = title.split()[:max_words]
    
    # Join words with underscores
    filename = '_'.join(words)
    
    # Remove or replace unsafe characters for filenames
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '')
    
    # Replace spaces and other problematic characters
    filename = filename.replace(' ', '_')
    filename = filename.replace('.', '_')
    filename = filename.replace(',', '_')
    filename = filename.replace('!', '')
    filename = filename.replace('?', '')
    filename = filename.replace("'", '')
    filename = filename.replace('"', '')
    
    # Remove multiple underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    # Ensure filename isn't empty
    if not filename:
        filename = 'untitled'
    
    # Limit length to avoid filesystem issues
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename

@app.route('/cleanup_temp_files', methods=['POST'])
def cleanup_temp_files():
    """Delete all temporary files created during video generation"""
    try:
        deleted_files = []
        
        # Patterns to match for temporary files
        temp_patterns = [
            'final_audio_*.wav',
            'reddit_post_*.png', 
            'temp_chunk_*.wav',
            'temp_hook_*.wav',
            'temp_story_by_*.wav',
            'subtitles_*.ass',
            '*_subtitle.wav'
        ]
        
        # Search for files matching each pattern
        for pattern in temp_patterns:
            matching_files = glob.glob(pattern)
            for file_path in matching_files:
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    logger.info(f"[CLEANUP] Deleted temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"[CLEANUP] Failed to delete {file_path}: {e}")
        
        # Clean up voice preview files
        if os.path.exists('temp_previews'):
            preview_files = os.listdir('temp_previews')
            for file in preview_files:
                try:
                    preview_path = os.path.join('temp_previews', file)
                    os.remove(preview_path)
                    deleted_files.append(preview_path)
                    logger.info(f"[CLEANUP] Deleted voice preview: {preview_path}")
                except Exception as e:
                    logger.warning(f"[CLEANUP] Failed to delete voice preview {file}: {e}")
        
        # Also clean up any orphaned temporary files in current directory
        all_files = os.listdir('.')
        for file in all_files:
            if any(keyword in file.lower() for keyword in ['final_audio', 'reddit_post', 'temp_chunk', 'temp_hook', 'temp_story', 'subtitles_', '_subtitle', 'voice_preview']):
                try:
                    os.remove(file)
                    deleted_files.append(file)
                    logger.info(f"[CLEANUP] Deleted orphaned temp file: {file}")
                except Exception as e:
                    logger.warning(f"[CLEANUP] Failed to delete {file}: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {len(deleted_files)} temporary files',
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Cleanup failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Cleanup failed: {str(e)}'
        }), 500

def extract_reddit_post_data(url):
    """Extract Reddit post data from URL"""
    try:
        if not validate_reddit_url(url):
            logger.error(f"[ERROR] Invalid Reddit URL: {url}")
            return None
        
        # Extract post ID and get post data
        post_id = extract_post_id_from_url(url)
        if not post_id:
            logger.error(f"[ERROR] Could not extract post ID from: {url}")
            return None
        
        # Scrape Reddit post
        logger.info(f"[DEBUG] Fetching Reddit post with ID: {post_id}")
        submission = reddit.submission(id=post_id)
        if submission.author is None:
            logger.error(f"[ERROR] Post not found or deleted: {post_id}")
            return None
        
        title = submission.title
        author = f"u/{submission.author.name}" if submission.author else "[deleted]"
        content = submission.selftext if submission.selftext else "[No text content]"
        
        logger.info(f"[DEBUG] Extracted post data:")
        logger.info(f"[DEBUG] Title: {title[:100]}...")
        logger.info(f"[DEBUG] Author: {author}")
        logger.info(f"[DEBUG] Content length: {len(content)} characters")
        
        return {
            'title': title,
            'author': author,
            'content': content
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to extract Reddit post data: {e}")
        return None

@app.route('/preview_hook', methods=['POST'])
def preview_hook():
    """Generate and preview hook with Reddit template"""
    try:
        data = request.get_json()
        logger.info(f"[INFO] Hook preview request received: {data}")
        
        mode = data.get('mode', 'auto')
        post_url = data.get('postUrl', '')
        
        if not post_url:
            return jsonify({'error': 'Post URL is required'}), 400
        
        # Extract Reddit post data
        logger.info(f"[INFO] Extracting Reddit post data from: {post_url}")
        post_data = extract_reddit_post_data(post_url)
        if not post_data:
            return jsonify({'error': 'Failed to extract Reddit post data'}), 500
        
        # Generate hook
        logger.info(f"[INFO] Generating hook for mode: {mode}")
        custom_hook_raw = data.get('customHook', '') if mode == 'user' else ''
        custom_hook = custom_hook_raw.strip() if custom_hook_raw else ''
        
        if mode == 'auto' or (mode == 'user' and not custom_hook):
            # Auto mode or User mode without custom hook - use AI generation
            if not llm:
                return jsonify({'error': 'LLaMA model not available for hook generation'}), 500
            hook = generate_tiktok_hook(post_data['title'], post_data['content'])
            hook_source = 'ai_generated'
        else:
            # User mode with custom hook provided
            hook = custom_hook
            hook_source = 'user_provided'
        
        if not hook:
            return jsonify({'error': 'Failed to generate hook'}), 500
        
        logger.info(f"[INFO] Generated hook: {hook[:100]}...")
        
        # Get user options for template generation
        user_options = None
        if mode == 'user':
            user_options = {
                'username': data.get('username', ''),
                'avatar': data.get('avatar', ''),
                'voice': data.get('voice', ''),
                'gameplay': data.get('gameplay', ''),
                'music': data.get('music', '')
            }
            logger.info(f"[DEBUG] User options for preview: {user_options}")
        
        # Generate Reddit post image preview
        username = user_options.get('username', settings.get('defaultUsername', 'OminousStories')) if user_options else settings.get('defaultUsername', 'OminousStories')
        avatar_path = user_options.get('avatar', get_default_media_file('avatar_images', 'OminousStoriesLogo.png')) if user_options else get_default_media_file('avatar_images', 'OminousStoriesLogo.png')
        
        logger.info(f"[INFO] Generating preview template with username: {username}")
        preview_image_path = generate_reddit_post_image(hook, username, avatar_path)
        
        if not preview_image_path:
            return jsonify({'error': 'Failed to generate preview template'}), 500
        
        # Return preview data
        response_data = {
            'hook': hook,
            'hook_source': hook_source,  # 'ai_generated' or 'user_provided'
            'preview_image': f'/preview_image/{preview_image_path}',
            'post_data': {
                'title': post_data['title'],
                'content': post_data['content'],
                'author': post_data['author']
            },
            'user_options': user_options
        }
        
        logger.info(f"[INFO] Hook preview generated successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to preview hook: {e}")
        return jsonify({'error': f'Failed to preview hook: {str(e)}'}), 500

@app.route('/regenerate_hook', methods=['POST'])
def regenerate_hook():
    """Regenerate hook for auto mode"""
    try:
        data = request.get_json()
        logger.info(f"[INFO] Hook regeneration request received")
        
        if not llm:
            return jsonify({'error': 'LLaMA model not available'}), 500
        
        title = data.get('title', '')
        content = data.get('content', '')
        
        if not title or not content:
            return jsonify({'error': 'Title and content are required'}), 400
        
        # Generate new hook
        logger.info(f"[INFO] Regenerating hook...")
        new_hook = generate_tiktok_hook(title, content)
        
        if not new_hook:
            return jsonify({'error': 'Failed to regenerate hook'}), 500
        
        logger.info(f"[INFO] New hook generated: {new_hook[:100]}...")
        
        # Get user options for template generation
        user_options = data.get('user_options')
        username = user_options.get('username', settings.get('defaultUsername', 'OminousStories')) if user_options else settings.get('defaultUsername', 'OminousStories')
        avatar_path = user_options.get('avatar', get_default_media_file('avatar_images', 'OminousStoriesLogo.png')) if user_options else get_default_media_file('avatar_images', 'OminousStoriesLogo.png')
        
        # Generate new preview template
        preview_image_path = generate_reddit_post_image(new_hook, username, avatar_path)
        
        if not preview_image_path:
            return jsonify({'error': 'Failed to generate new preview template'}), 500
        
        response_data = {
            'hook': new_hook,
            'preview_image': f'/preview_image/{preview_image_path}'
        }
        
        logger.info(f"[INFO] Hook regenerated successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to regenerate hook: {e}")
        return jsonify({'error': f'Failed to regenerate hook: {str(e)}'}), 500

@app.route('/proceed_with_video', methods=['POST'])
def proceed_with_video():
    """Proceed with full video generation after hook approval"""
    try:
        data = request.get_json()
        logger.info(f"[INFO] Proceeding with video generation after hook approval")
        
        mode = data.get('mode', 'auto')
        approved_hook = data.get('approved_hook', '')
        post_data = data.get('post_data', {})
        user_options = data.get('user_options')
        
        if not approved_hook or not post_data:
            return jsonify({'error': 'Approved hook and post data are required'}), 400
        
        # Add the approved hook to post data
        post_data['hook'] = approved_hook
        
        logger.info(f"[INFO] Starting video generation with approved hook: {approved_hook[:50]}...")
        
        # Validate user options for problematic values if in user mode
        if mode == 'user' and user_options:
            logger.info(f"[DEBUG] User options for video: {user_options}")
            
            # Validate user options for problematic values
            for key, value in user_options.items():
                if value is not None and str(value).strip() == '':
                    logger.warning(f"[WARNING] Empty user option detected: {key} = '{value}', this may cause issues")
        
        # Start video generation process
        process_id = str(uuid.uuid4())
        
        # Start background thread for video processing
        thread = threading.Thread(
            target=process_video_generation,
            args=(process_id, post_data, mode, user_options)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'process_id': process_id,
            'message': 'Video generation started with approved hook'
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to proceed with video generation: {e}")
        return jsonify({'error': f'Failed to start video generation: {str(e)}'}), 500

@app.route('/preview_image/<filename>')
def serve_preview_image(filename):
    """Serve preview images"""
    try:
        return send_file(filename, mimetype='image/png')
    except FileNotFoundError:
        return "Preview image not found", 404

@app.route('/preview_voice', methods=['POST'])
def preview_voice():
    """Generate a voice preview sample"""
    try:
        data = request.get_json()
        voice_id = data.get('voice_id')
        
        if not voice_id:
            return jsonify({'success': False, 'error': 'Voice ID required'})
        
        # Sample text for voice preview
        sample_text = "Hello! This is a preview of this voice. Here's how your videos will sound."
        
        # Generate temporary audio file
        preview_filename = f"voice_preview_{voice_id}_{int(time.time())}.wav"
        preview_path = f"temp_previews/{preview_filename}"
        
        # Ensure temp previews directory exists
        os.makedirs("temp_previews", exist_ok=True)
        
        # Generate TTS preview
        if generate_tts_audio(sample_text, voice_id, preview_path):
            return jsonify({
                'success': True, 
                'preview_url': f'/voice_preview/{preview_filename}',
                'filename': preview_filename
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to generate voice preview'})
            
    except Exception as e:
        logger.error(f"Voice preview error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/voice_preview/<filename>')
def serve_voice_preview(filename):
    """Serve voice preview audio file"""
    try:
        preview_path = os.path.join("temp_previews", filename)
        if os.path.exists(preview_path):
            return send_file(preview_path, as_attachment=False, mimetype='audio/wav')
        else:
            return "Preview not found", 404
    except Exception as e:
        logger.error(f"Error serving voice preview: {e}")
        return "Error serving preview", 500

@app.route('/cleanup_voice_preview', methods=['POST'])
def cleanup_voice_preview():
    """Clean up voice preview file after use"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if filename:
            preview_path = os.path.join("temp_previews", filename)
            if os.path.exists(preview_path):
                os.unlink(preview_path)
                logger.info(f"Cleaned up voice preview: {filename}")
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Voice preview cleanup error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    logger.info("Starting Flask application initialization...")
    
    logger.info("Ensuring required folders exist...")
    ensure_folders()
    
    logger.info("Loading settings from file...")
    load_settings_from_file()
    
    logger.info("System Status:")
    logger.info(f"  - LLaMA Model: {'Loaded' if llm else 'Not Available'}")
    logger.info(f"  - TTS Model: {'Loaded' if tts else 'Not Available'}")
    logger.info(f"  - Whisper Model: {'Loaded' if whisper_model else 'Not Available'}")
    logger.info(f"  - Video Processing: {'Available' if VIDEO_PROCESSING_AVAILABLE else 'Not Available'}")
    logger.info(f"  - Reddit API: Connected")
    
    logger.info("Starting Flask web server...")
    logger.info("Server will be available at: http://127.0.0.1:5000")
    logger.info("Debug mode: Enabled")
    logger.info("TikTok Video Generator is ready!")
    
    # Advanced Windows console encoding fixes
    if sys.platform == "win32":
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Try multiple fallback approaches for Windows console issues
        try:
            # First attempt: Safe stream redirection
            import contextlib
            import io
            
            safe_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            safe_stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            
            with contextlib.redirect_stdout(safe_stdout), contextlib.redirect_stderr(safe_stderr):
                app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
        except Exception as e:
            logger.warning(f"Console redirection failed: {e}")
            try:
                # Second attempt: Disable debug mode
                logger.info("Starting Flask with minimal console output...")
                app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
            except Exception as e2:
                logger.error(f"Flask startup failed: {e2}")
                try:
                    # Final attempt: Suppress all output
                    logger.info("Starting Flask with suppressed output...")
                    import os
                    with open(os.devnull, 'w') as devnull:
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        sys.stdout = devnull
                        sys.stderr = devnull
                        try:
                            app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
                        finally:
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr
                except Exception as e3:
                    logger.error(f"All Flask startup methods failed: {e3}")
                    logger.info("Check if Flask started at http://127.0.0.1:5000")
    else:
        # Non-Windows systems
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False) 
