# START OF FILE config.py

"""
Configuration File for Nemai Discord Bot

This file centralizes all configuration settings for the bot.
It loads sensitive information (like API keys and tokens) from environment
variables using `python-dotenv` and defines constants for various bot
parameters, including database settings, model names, limits, logging,
and UI elements.
"""

import logging
import os
import discord
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists.
# This is useful for development environments.
load_dotenv()

# --- API Keys and Tokens ---
# Essential credentials loaded from environment variables.
# The bot will fail to start if DISCORD_TOKEN or GEMINI_API_KEY are missing.
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN") # Your Discord Bot Token
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Your Google AI (Gemini) API Key
ADMIN_USER_ID_STR = os.getenv("ADMIN_USER_ID") # Discord User ID of the bot administrator (optional)
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN') # Hugging Face API Token (optional, for /imagine)

# --- Admin Configuration ---
# Convert the Admin User ID string to an integer if provided.
ADMIN_USER_ID = None
if ADMIN_USER_ID_STR:
    try:
        ADMIN_USER_ID = int(ADMIN_USER_ID_STR)
    except ValueError:
        # Log a warning if the ID is invalid. Admin commands will be disabled.
        print("Warning: ADMIN_USER_ID environment variable is not a valid integer. Admin commands disabled.")

# --- Bot Settings ---
BOT_PREFIX = "?" # Prefix for legacy text-based admin commands (e.g., ?admin backup)

# --- Database Settings ---
DATABASE_NAME = 'nemai_bot.db' # Filename for the SQLite database
MAX_PERSONAS = 5 # Maximum number of custom personas a user can create
# List of valid identifiers for different conversation types stored in the database.
VALID_CONVERSATION_TYPES = ["chat", "sherlock", "teacher", "scientist", "persona", "doc_assist", "recommend", "story"]
PRUNE_DAYS = 30 # Number of days after which old conversation history will be pruned
BACKUP_DIR = "backups" # Directory to store database backups

# --- History and Interaction Limits ---
HISTORY_LIMIT_DEFAULT = 20 # Default number of messages included in standard history context
RELEVANT_HISTORY_LIMIT_DEFAULT = 10 # Default number of messages to retrieve for relevant history context
SIMILARITY_THRESHOLD_DEFAULT = 0.6 # Default cosine similarity threshold for relevant history/feedback
SENTIMENT_LIMIT_DEFAULT = 50 # Default number of messages to analyze for /sentiment_stats
SENTIMENT_LIMIT_MAX = 200 # Maximum number of messages allowed for /sentiment_stats
SENTIMENT_NEUTRAL_THRESHOLD_LOWER = 0.4 # Lower bound for classifying sentiment as neutral (not currently used)
SENTIMENT_NEUTRAL_THRESHOLD_UPPER = 0.6 # Upper bound for classifying sentiment as neutral (not currently used)
EXPORT_LIMIT_DEFAULT = 50 # Default number of messages to include in /export
EXPORT_LIMIT_MAX = 500 # Maximum number of messages allowed for /export
HISTORY_SEARCH_RESULT_LIMIT = 10 # Default number of results for /search_history
RECOMMENDATION_LIMIT = 5 # Number of recommendations to provide with /recommend get
MAX_USER_PREFERENCES = 200 # Max character length for user preference strings (likes/dislikes)
STORY_CONTEXT_LIMIT = 10 # Max number of recent turns to include in story prompts
STORY_TIMEOUT_MINUTES = 60 # Minutes after which an inactive story session is considered timed out

# --- AI Model Configuration ---
# Names of models used for embeddings and generation.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Sentence Transformer model for embeddings
MODEL_CHAT_NAME = 'gemini-2.0-flash' # Gemini model for general chat, summaries, etc.
MODEL_PERSONA_NAME = 'gemini-2.0-flash' # Gemini model for custom personas and built-in ones like Sherlock
MODEL_VISION_NAME = 'gemini-2.0-flash' # Gemini model supporting vision (image analysis)
# Generation configurations for different Gemini models (e.g., temperature controls creativity).
MODEL_CONFIG_FLASH = {"temperature": 0.8}
MODEL_CONFIG_PERSONA = {"temperature": 1.25} # Higher temperature for more creative persona responses
MODEL_CONFIG_VISION = {"temperature": 0.6}
MODEL_CONFIG_DEBATE = {"temperature": 1.35} # Higher temperature for more distinct debate arguments
# Hugging Face Inference API endpoint for image generation (/imagine command).
IMAGE_GEN_MODEL_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0" # SDXL Base 1.0

# --- Logging Configuration ---
LOG_FILE = 'nemai_bot.log' # Filename for the log file
LOG_LEVEL = logging.INFO # Logging level (e.g., INFO, DEBUG, WARNING)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Format for log messages
LOG_ROTATION_MAX_BYTES = 5 * 1024 * 1024 # 5 MB max size before log rotation
LOG_ROTATION_BACKUP_COUNT = 3 # Number of backup log files to keep

# --- Discord UI and Message Limits ---
MAX_EMBED_DESCRIPTION = 4096 # Discord's maximum character limit for embed descriptions
MAX_PAGE_SIZE = 3800 # Maximum characters per page in the paginator (slightly less than embed limit for safety)
PAGINATOR_TIMEOUT = 180.0 # Seconds before paginator buttons become disabled (3 minutes)
# Color definitions for embeds used in various commands.
EMBED_COLORS = {
    "default": discord.Color.blue(),
    "citation": discord.Color.gold(),
    "comparison": discord.Color.green(),
    "definition": discord.Color.purple(),
    "pros_cons": discord.Color.teal(),
    "sherlock": discord.Color.dark_orange(),
    "persona_custom": discord.Color.orange(),
    "persona_debate": discord.Color.dark_magenta(),
    "error": discord.Color.red(),
    "admin": discord.Color.dark_red(),
    "success": discord.Color.green(),
    "info": discord.Color.light_grey(),
    "warning": discord.Color.orange(),
    "user_stats": discord.Color.blue(),
    "persona_list": discord.Color.purple(),
    "image": discord.Color.magenta(), # For /analyze_file image results
    "imagine": discord.Color.from_rgb(255, 192, 203), # Pink for /imagine results
    "search": discord.Color.dark_grey(),
    "sentiment_stats": discord.Color.from_rgb(255, 105, 180), # Pink
    "news": discord.Color.dark_blue(),
    "factcheck": discord.Color.dark_green(),
    "history_search": discord.Color.dark_gold(),
    "doc_assist": discord.Color.dark_teal(),
    "recommend": discord.Color.dark_gold(),
    "story": discord.Color.dark_purple(),
    "summary": discord.Color.from_rgb(135, 206, 250), # Light Sky Blue
    "recipe": discord.Color.from_rgb(210, 180, 140), # Tan
}

# --- Feature-Specific Limits and Settings ---
# Persona creation limits.
PERSONA_NAME_MIN_LEN = 2
PERSONA_NAME_MAX_LEN = 30
PERSONA_DESC_MIN_LEN = 10
PERSONA_DESC_MAX_LEN = 1500
# Search and news result limits.
SEARCH_RESULT_LIMIT = 8 # Max web results fetched for /search, /factcheck, etc.
NEWS_ARTICLE_LIMIT = 8 # Max news articles fetched for /news
# Document assistance limits.
DOC_ASSIST_MAX_SECTION_LEN = 2000 # Max characters for /doc_assist rewrite_section input
# Image generation timeout.
IMAGE_GEN_TIMEOUT_SECONDS = 90 # Max time to wait for Hugging Face API response

# --- File Handling Settings ---
# Allowed MIME types for file analysis (/analyze_file).
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/heic", "image/heif"]
ALLOWED_TEXT_TYPES = [
    "text/plain", "text/markdown", "text/csv", "text/html", "text/xml",
    "application/json", "application/x-python", "application/javascript",
    "text/x-log", "text/css", "text/x-c", "text/x-c++src", "text/x-java-source",
    "text/x-shellscript", "text/x-php", "text/x-ruby",
    "application/pdf", "application/msword", # .pdf, .doc
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
    "application/vnd.oasis.opendocument.text" # .odt
]
# Combined list for validation.
ALL_ALLOWED_FILE_TYPES = ALLOWED_IMAGE_TYPES + ALLOWED_TEXT_TYPES
# File size and content limits for analysis.
MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024 # 8 MB maximum file upload size
MAX_FILE_CONTENT_CHARS = 50000 # Max characters extracted from text files for analysis

# --- Feedback System Configuration ---
ENABLE_FEEDBACK_SYSTEM = True # Toggle the feedback (👍/👎) system on/off
MAX_POSITIVE_EXAMPLES_IN_PROMPT = 3 # Max positive feedback examples to include in AI prompts
MAX_NEGATIVE_EXAMPLES_IN_PROMPT = 2 # Max negative feedback examples to include in AI prompts
FEEDBACK_LOOKBACK_DAYS = 14 # How many days back to search for relevant feedback examples
FEEDBACK_SIMILARITY_THRESHOLD = 0.65 # Minimum similarity score for feedback to be considered relevant

# --- Startup Validation ---
# Ensure critical environment variables are set.
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable not set.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
# Warn if the optional Hugging Face token is missing, as /imagine will be disabled.
if not HUGGINGFACE_TOKEN:
    print("Warning: HUGGINGFACE_TOKEN environment variable not set. /imagine command will be disabled.")

# END OF FILE config.py