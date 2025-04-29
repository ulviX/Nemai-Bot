# START OF FILE main.py

"""
Main Entry Point for the Nemai Discord Bot

This script initializes the Discord client, loads necessary AI models and pipelines,
connects to the database, sets up command handlers, and runs the bot.
It also includes event handlers for `on_ready`, `on_message` (for admin commands),
and `on_interaction` (primarily for handling button clicks from UI views like StoryChoiceView).
A periodic maintenance task is scheduled to perform database backups and pruning.
"""

import discord
from discord import app_commands
import google.generativeai as genai
import asyncio
import datetime
import io
import logging
import os
import numpy as np
from logging.handlers import RotatingFileHandler
from database import Database, embedding_model as db_embedding_model # Import DB and potentially reuse its embedding model
from paginator import PaginatorView, chunk_message
import config # Bot configuration settings
from transformers import pipeline # For sentiment analysis
import commands # Module containing command implementations and setup
from commands import FeedbackView, StoryChoiceView # Import specific UI views for registration

# --- Configuration Validation ---
# Ensure essential API keys are present in the configuration.
if not config.DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN configuration missing.")
if not config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY configuration missing.")
# Warn if Admin User ID is set but invalid, or not set at all.
if config.ADMIN_USER_ID is None and config.ADMIN_USER_ID_STR is not None:
     print("Warning: ADMIN_USER_ID configured incorrectly. Admin commands will not work.")
elif config.ADMIN_USER_ID is None:
     print("Warning: ADMIN_USER_ID not configured. Admin commands will not work.")

# --- Logging Setup ---
# Configure the root logger.
root_logger = logging.getLogger()
root_logger.setLevel(config.LOG_LEVEL)

# Clear existing handlers to prevent duplicate logging if the script is reloaded.
if root_logger.handlers:
    root_logger.handlers.clear()

# Create a formatter based on the config.
formatter = logging.Formatter(config.LOG_FORMAT)

# Create a rotating file handler to manage log file size.
file_handler = RotatingFileHandler(
    config.LOG_FILE,
    maxBytes=config.LOG_ROTATION_MAX_BYTES,
    backupCount=config.LOG_ROTATION_BACKUP_COUNT,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Create a stream handler to output logs to the console.
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

# Get a specific logger instance for this main module.
logger = logging.getLogger('nemai_main')
# Set higher logging levels for noisy libraries to keep the main log cleaner.
logging.getLogger('discord').setLevel(logging.WARNING)
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('PyPDF2').setLevel(logging.WARNING)
logging.getLogger('docx').setLevel(logging.WARNING)

# --- Model Initialization ---
# Initialize global variables for AI models and pipelines.
model_chat = None
model_persona = None
model_vision = None
model_debate = None
sentiment_pipeline = None
embedding_model = None

# Initialize Gemini Models
try:
    genai.configure(api_key=config.GEMINI_API_KEY)
    # Load models specified in the config with their respective generation settings.
    model_chat = genai.GenerativeModel(model_name=config.MODEL_CHAT_NAME, generation_config=config.MODEL_CONFIG_FLASH)
    model_persona = genai.GenerativeModel(model_name=config.MODEL_PERSONA_NAME, generation_config=config.MODEL_CONFIG_PERSONA)
    model_vision = genai.GenerativeModel(model_name=config.MODEL_VISION_NAME, generation_config=config.MODEL_CONFIG_VISION)
    model_debate = genai.GenerativeModel(model_name=config.MODEL_PERSONA_NAME, generation_config=config.MODEL_CONFIG_DEBATE) # Using persona model with debate config
    logger.info("Gemini AI models configured successfully.")
except Exception as e:
    # Log critical error if Gemini setup fails. Bot might not function correctly.
    logger.critical(f"Failed to configure Gemini AI: {e}", exc_info=True)

# Initialize Hugging Face Sentiment Analysis Pipeline
try:
    # Load the specified sentiment analysis model.
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    logger.info("Transformers sentiment analysis pipeline initialized successfully.")
except Exception as e:
    # Log error if pipeline fails. Sentiment analysis features will be unavailable.
    logger.error(f"Failed to initialize Transformers sentiment analysis pipeline: {e}", exc_info=True)

# Initialize Embedding Model (Sentence Transformer)
try:
    # Check if the model was already loaded by the database module.
    if db_embedding_model:
        embedding_model = db_embedding_model
        logger.info("Reusing Sentence Transformer model from database module.")
    else:
        # If not loaded by DB (shouldn't happen if DB init succeeded), raise error.
        raise ImportError("Embedding model not loaded in database module.")
except ImportError as e:
    # Fallback: Attempt to load the model separately if reuse failed.
    logger.warning(f"Could not reuse embedding model from database: {e}. Attempting separate load.")
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        logger.info(f"Successfully loaded separate Sentence Transformer model: {config.EMBEDDING_MODEL_NAME}")
    except Exception as e:
        # Log critical error if embedding model fails to load. Core features will break.
        logger.critical(f"Failed to load Sentence Transformer model '{config.EMBEDDING_MODEL_NAME}' in main: {e}", exc_info=True)
        embedding_model = None # Ensure it's None if loading fails.

# --- Discord Client Setup ---
# Define necessary intents for the bot.
intents = discord.Intents.default()
intents.message_content = True # Required to read message content.
intents.members = True # Required for user information like display names, potentially roles.
intents.presences = False # Presences are usually not needed and require privileged intent.

# Initialize Database
db = None
try:
    db = Database() # Instantiate the database class.
except Exception as e:
     # If database fails to initialize (e.g., model loading error), log and exit.
     logger.critical(f"Failed to initialize database: {e}. Bot cannot start.", exc_info=True)
     exit()

# Initialize Discord Client and Command Tree
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client) # Command tree for slash commands.

# Define Application Command Groups (for organizing slash commands like /persona create)
persona_group = app_commands.Group(
    name="persona",
    description=f"Create and manage custom AI personas (Slots 1-{config.MAX_PERSONAS})"
)
doc_assist_group = app_commands.Group(
    name="doc_assist",
    description="Get AI assistance with writing documents."
)
recommend_group = app_commands.Group(
    name="recommend",
    description="Get and manage media recommendations."
)
story_group = app_commands.Group(
    name="story",
    description="Create and participate in interactive stories."
)

# --- Event Handlers ---

@client.event
async def on_ready():
    """
    Event handler called when the bot successfully connects to Discord and is ready.

    Initializes command handlers, schedules maintenance tasks, syncs application
    commands with Discord, and sets the bot's presence.
    """
    logger.info(f'Logged in as {client.user} (ID: {client.user.id})')
    logger.info('------')

    # Setup command handlers by passing necessary objects to the commands module.
    commands.setup_commands(
        _db=db,
        _client=client,
        _tree=tree,
        _persona_group=persona_group,
        _doc_assist_group=doc_assist_group,
        _recommend_group=recommend_group,
        _story_group=story_group,
        _model_chat=model_chat,
        _model_persona=model_persona,
        _model_vision=model_vision,
        _model_debate=model_debate,
        _sentiment_pipeline=sentiment_pipeline,
        _embedding_model=embedding_model,
        _logger=logger # Pass the main logger instance
    )

    # Start the background task for periodic database maintenance.
    client.loop.create_task(periodic_maintenance())
    logger.info("Periodic maintenance task scheduled.")

    try:
        # Register persistent views (like FeedbackView) so they work after bot restarts.
        if config.ENABLE_FEEDBACK_SYSTEM:
            # Pass dummy values; actual values are set when the view is used.
            # The custom_id structure allows linking back to the specific interaction.
            client.add_view(FeedbackView(0, 0, "", "", None))
            logger.info("Registered persistent FeedbackView.")

        # StoryChoiceView is typically created per-message and doesn't need persistence
        # unless story sessions need to survive restarts AND retain interactive buttons.
        # If persistence is needed, the view/callbacks would need logic to re-fetch
        # session state based on custom_id components (session_id, user_id).
        # client.add_view(StoryChoiceView(0, "", [])) # Example placeholder if persistence was needed

        # Sync application (slash) commands with Discord.
        # This registers/updates the commands users see.
        synced = await tree.sync()
        logger.info(f"Synced {len(synced)} global application commands.")

        if config.ENABLE_FEEDBACK_SYSTEM:
             logger.info("Feedback system enabled. New messages will have feedback buttons.")

    except Exception as e:
        logger.exception(f"Failed to sync application commands or register views: {e}")

    # Set the bot's activity status (e.g., "Playing /help").
    await client.change_presence(activity=discord.Game(name="/help"))


async def periodic_maintenance():
    """
    Background task performing scheduled maintenance (backup, pruning).
    """
    await client.wait_until_ready() # Ensure the bot is ready before starting.
    logger.info("Periodic maintenance task started.")
    while not client.is_closed():
        now = datetime.datetime.now(datetime.timezone.utc)
        try:
            # --- Weekly Backup ---
            # Check if it's Sunday 3 AM UTC.
            if now.weekday() == 6 and now.hour == 3: # Sunday = 6
                logger.info("Performing weekly database backup...")
                backup_filename = db.backup_database()
                if backup_filename:
                    logger.info(f"Database backup successful: {backup_filename}")
                else:
                    logger.error("Database backup failed.")
                # Sleep for an hour after backup to avoid immediate re-check.
                await asyncio.sleep(3600)
                continue # Skip the rest of the loop for this hour.

            # --- Monthly Pruning ---
            # Check if it's the 1st day of the month, 4 AM UTC.
            if now.day == 1 and now.hour == 4:
                 logger.info(f"Performing monthly conversation pruning (older than {config.PRUNE_DAYS} days)...")
                 deleted_count = db.prune_old_conversations(days=config.PRUNE_DAYS)
                 logger.info(f"Pruning complete. Deleted {deleted_count} old conversations.")
                 # Optional: Add pruning for inactive story sessions here if desired.
                 # inactive_story_prune = db.prune_inactive_stories(config.PRUNE_DAYS)
                 # logger.info(f"Pruned {inactive_story_prune} inactive story sessions.")
                 # Sleep for an hour after pruning.
                 await asyncio.sleep(3600)
                 continue # Skip the rest of the loop.

            # If no maintenance task ran, wait for an hour before checking again.
            await asyncio.sleep(3600) # Check hourly

        except Exception as e:
            logger.exception(f"Error during periodic maintenance: {e}")
            # Wait longer after an error to prevent log spam if the error persists.
            await asyncio.sleep(3600 * 3) # Sleep for 3 hours


@client.event
async def on_message(message: discord.Message):
    """
    Event handler called when a message is sent in a channel the bot can see.

    Handles legacy text-based admin commands (prefixed with `config.BOT_PREFIX`).
    Ignores messages from bots or those not starting with the admin prefix.
    """
    # Ignore messages from bots and those not starting with the defined prefix.
    if message.author.bot or not message.content.startswith(config.BOT_PREFIX):
        return

    # Check specifically for admin commands.
    if message.content.startswith(f"{config.BOT_PREFIX}admin"):
        # Verify if the author is the configured admin user.
        if config.ADMIN_USER_ID and message.author.id == config.ADMIN_USER_ID:
            logger.info(f"Admin command received from {message.author}: {message.content}")
            # Parse the command parts.
            admin_cmd_parts = message.content.split()
            command = admin_cmd_parts[1].lower() if len(admin_cmd_parts) > 1 else None

            # --- Admin Command Handling ---
            if command == "backup":
                try:
                    await message.channel.send("⏳ Starting database backup...")
                    backup_file = db.backup_database()
                    if backup_file:
                        await message.channel.send(f"✅ Database backed up successfully to `{os.path.basename(backup_file)}`")
                    else:
                        await message.channel.send("❌ Database backup failed. Check logs.")
                except Exception as e:
                     logger.exception("Admin backup command failed.")
                     await message.channel.send(f"❌ Backup error: {e}")
                return # Stop further processing

            elif command == "prune":
                # Expecting "?admin prune [days]"
                if len(admin_cmd_parts) > 2:
                    try:
                        days = int(admin_cmd_parts[2])
                        if days < 0: raise ValueError("Days must be positive.")
                        await message.channel.send(f"⏳ Pruning messages older than {days} days...")
                        count = db.prune_old_conversations(days)
                        await message.channel.send(f"✅ Pruned {count} messages older than {days} days and updated user stats.")
                    except ValueError:
                        await message.channel.send(f"❌ Invalid number of days. Usage: `{config.BOT_PREFIX}admin prune [positive_number_of_days]`")
                    except Exception as e:
                        logger.exception("Admin prune command failed.")
                        await message.channel.send(f"❌ Pruning error: {e}")
                else:
                     # Provide usage instructions if days argument is missing.
                     await message.channel.send(f"Usage: `{config.BOT_PREFIX}admin prune [days]`")
                return # Stop further processing

            elif command == "stats":
                 try:
                    admin_stats = db.get_admin_stats()
                    embed = discord.Embed(title="⚙️ Admin Bot Statistics", color=config.EMBED_COLORS.get("admin", discord.Color.dark_red()))
                    embed.add_field(name="Total Users", value=admin_stats.get('total_users', 'N/A'))
                    embed.add_field(name="Total Messages (Saved)", value=admin_stats.get('total_messages', 'N/A'))
                    embed.add_field(name="Total Personas", value=admin_stats.get('total_personas', 'N/A'))
                    embed.add_field(name="Total Feedback", value=admin_stats.get('total_feedback', 'N/A'))
                    embed.add_field(name="Total Preferences", value=admin_stats.get('total_preferences', 'N/A'))
                    embed.add_field(name="Active Stories", value=admin_stats.get('active_stories', 'N/A'))
                    embed.add_field(name="DB Size (MB)", value=f"{admin_stats.get('database_size_mb', 'N/A')} MB")
                    await message.channel.send(embed=embed)
                 except Exception as e:
                      logger.exception("Admin stats command failed.")
                      await message.channel.send(f"❌ Error getting admin stats: {e}")
                 return # Stop further processing

            elif command == "shutdown":
                logger.warning(f"Shutdown command received from admin {message.author}.")
                await message.channel.send("⚠️ Shutting down bot...")
                await client.close() # Gracefully close the bot connection.
                return # Stop further processing

            else:
                # Handle unknown admin commands.
                await message.channel.send(f"Unknown admin command. Available: `backup`, `prune [days]`, `stats`, `shutdown` (Prefix: `{config.BOT_PREFIX}admin`)")
                return # Stop further processing
        else:
            # Handle unauthorized admin command attempts.
            logger.warning(f"Unauthorized admin command attempt by {message.author}: {message.content}")
            try:
               # Attempt to delete the unauthorized command message.
               await message.delete()
               # Send a temporary message informing the user.
               await message.channel.send("You do not have permission to use admin commands.", delete_after=10)
            except discord.Forbidden:
                 logger.warning("Missing permissions to delete unauthorized admin command message.")
            except discord.NotFound:
                 pass # Message might have been deleted already.
            return # Stop further processing

@client.event
async def on_interaction(interaction: discord.Interaction):
    """
    Event handler called when an interaction (e.g., slash command, button click) occurs.

    Currently handles button clicks specifically for the StoryChoiceView.
    Slash commands are handled automatically by the `app_commands.CommandTree`.
    """
    # Check if the interaction is a component interaction (button, select menu, etc.)
    # and if the custom_id matches the pattern for story choice buttons.
    if interaction.type == discord.InteractionType.component and interaction.data.get('custom_id', '').startswith("story_choice_"):
        custom_id = interaction.data['custom_id']
        custom_id_parts = custom_id.split('_')
        # Validate the custom_id format. Expecting "story_choice_{session_id}_{index}"
        if len(custom_id_parts) < 4:
            logger.warning(f"Invalid story choice custom_id format: {custom_id}")
            await interaction.response.send_message("Invalid choice button.", ephemeral=True)
            return

        try:
            # Extract the session ID from the custom_id.
            session_id = int(custom_id_parts[2])
            # Note: The choice index (custom_id_parts[3]) isn't strictly needed here
            # because the callback itself knows the choice text.
        except ValueError:
            logger.error(f"Could not parse session ID from story choice custom_id: {custom_id}")
            await interaction.response.send_message("Error processing choice.", ephemeral=True)
            return

        # The actual processing logic is handled within the StoryChoiceView's button callback
        # which calls `story_process_choice_impl`. This event handler primarily exists
        # to acknowledge the interaction type if more complex routing were needed,
        # or potentially to re-fetch the view if it wasn't persistent (though StoryChoiceView is designed per-message).

        # Example of how you might find the view if needed (less relevant now as callbacks handle it):
        # user_id = str(interaction.user.id)
        # view = next((v for v in client.persistent_views if isinstance(v, StoryChoiceView) and v.session_id == session_id and v.user_id == user_id), None)
        # if view:
        #     # View found, its callback will handle the logic.
        #     pass
        # else:
        #     # View not found (e.g., timed out, bot restarted without persistence)
        #     logger.warning(f"Could not find matching StoryChoiceView for interaction {interaction.id}, custom_id {custom_id}")
        #     # Respond to the interaction to prevent "Interaction failed"
        #     try:
        #         await interaction.response.send_message("This story choice is no longer active.", ephemeral=True)
        #     except discord.InteractionResponded:
        #         pass # Already responded to (likely by the view's timeout or callback deferral)
        pass # Let the view's callback handle the interaction.


# --- Main Execution Block ---
if __name__ == "__main__":
    # Perform final checks before starting the bot.
    if not config.DISCORD_TOKEN:
        logger.critical("Bot cannot start: DISCORD_TOKEN is not set in environment variables or config.")
    elif not config.GEMINI_API_KEY:
         logger.critical("Bot cannot start: GEMINI_API_KEY is not set in environment variables or config.")
    elif db is None:
         logger.critical("Bot cannot start: Database initialization failed.")
    elif model_chat is None or model_persona is None or model_vision is None or model_debate is None:
         logger.critical("Bot cannot start: Failed to initialize one or more AI models.")
    elif embedding_model is None:
         logger.critical("Bot cannot start: Failed to initialize Sentence Transformer model.")
    elif sentiment_pipeline is None:
         logger.warning("Bot starting without sentiment analysis pipeline. /sentiment_stats will be unavailable.") # Warning, not critical

    # Aggregate startup checks.
    startup_ok = (
        config.DISCORD_TOKEN and
        config.GEMINI_API_KEY and
        db is not None and
        model_chat is not None and
        model_persona is not None and
        model_vision is not None and
        model_debate is not None and
        embedding_model is not None
        # sentiment_pipeline is optional for basic startup
    )

    if startup_ok:
        try:
            logger.info("Starting bot...")
            # Run the Discord client with the token.
            # `log_handler=None` prevents discord.py from setting up its own root logger handler,
            # as we've already configured it.
            client.run(config.DISCORD_TOKEN, log_handler=None)
        except discord.LoginFailure:
            logger.critical("Login failed: Invalid Discord token.")
        except discord.PrivilegedIntentsRequired:
             logger.critical("Intents error: Ensure Message Content and Server Members intents are enabled in the Discord Developer Portal.")
        except Exception as e:
            # Catch any other fatal errors during runtime.
            logger.critical(f"Fatal error running bot: {e}", exc_info=True)
        finally:
            # --- Cleanup ---
            # Ensure database connection is closed gracefully on exit.
            if db and hasattr(db, 'conn') and db.conn:
                 logger.info("Ensuring database connection is closed in final cleanup.")
                 try:
                     db.close()
                 except Exception as db_close_err:
                     logger.error(f"Error closing database connection during final cleanup: {db_close_err}", exc_info=True)
            logger.info("Bot process finished.")
    else:
        # Log if startup checks failed.
        logger.critical("Bot prevented from starting due to initialization errors (Database, AI Models, Embedding Model, or API Keys). Check logs.")

# END OF FILE main.py