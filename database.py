# START OF FILE database.py

import sqlite3
import datetime
import logging
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import config
import json
from operator import itemgetter
from typing import Optional, Union, List, Dict, Any, Tuple
# --- Global Variables ---
embedding_model = None # Holds the loaded sentence transformer model.
EMBEDDING_DIM = None   # Stores the dimension of the embeddings produced by the model.

# --- Model Loading ---
# Attempt to load the Sentence Transformer model specified in the config.
# This is crucial for embedding generation and similarity searches.
try:
    # Load the model using the name from the configuration file.
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    # Get the embedding dimension once the model is loaded.
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    logging.info(f"Successfully loaded SentenceTransformer model '{config.EMBEDDING_MODEL_NAME}' with dimension {EMBEDDING_DIM}.")
except Exception as e:
    # Log a critical error if the model fails to load, as the database relies on it.
    logging.critical(f"Failed to load SentenceTransformer model '{config.EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
    # embedding_model remains None, which will be checked in the Database constructor.

# --- Logger Setup ---
# Configure logging specifically for the database module.
logger = logging.getLogger('database')
logger.setLevel(config.LOG_LEVEL) # Set level from config.

# Clear existing handlers to avoid duplicate logs if reloaded.
if logger.handlers:
    logger.handlers.clear()

# Define log format.
formatter = logging.Formatter(config.LOG_FORMAT)

# Add file handler for persistent logging.
file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Add stream handler for console output during development/debugging.
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Prevent logs from propagating to the root logger if handlers are configured here.
logger.propagate = False

# --- Database Class ---

class Database:
    """
    Handles all interactions with the SQLite database for the Nemai bot.

    This class manages user data, conversation history, custom personas,
    feedback, preferences, story sessions, and associated embeddings.
    It provides methods for creating, reading, updating, and deleting data,
    as well as performing semantic searches on conversation history.

    Attributes:
        MAX_PERSONAS (int): Maximum number of custom personas allowed per user (from config).
        VALID_CONVERSATION_TYPES (list): List of valid conversation type identifiers (from config).
        RELEVANCE_FETCH_POOL_SIZE (int): Number of recent messages to fetch initially for relevance calculation.
        RELEVANCE_GUARANTEED_RECENT (int): Number of most recent messages always included in relevant history.
        conn (sqlite3.Connection): The connection object to the SQLite database.
        cursor (sqlite3.Cursor): The cursor object for executing SQL queries.
    """
    # Constants loaded from config for easy access within the class.
    MAX_PERSONAS = config.MAX_PERSONAS
    VALID_CONVERSATION_TYPES = config.VALID_CONVERSATION_TYPES
    RELEVANCE_FETCH_POOL_SIZE = 30 # How many messages to initially pull for relevance check
    RELEVANCE_GUARANTEED_RECENT = 5 # How many *most recent* messages are always included

    def __init__(self, db_name=config.DATABASE_NAME):
        """
        Initializes the Database connection and sets up necessary tables.

        Args:
            db_name (str): The path to the SQLite database file. Defaults to the value in config.

        Raises:
            RuntimeError: If the Sentence Transformer embedding model failed to load.
            sqlite3.Error: If there's an error connecting to or setting up the database.
        """
        # Critical check: Ensure the embedding model loaded successfully.
        if embedding_model is None:
             logger.critical("Database cannot be initialized: Sentence Transformer model is not available.")
             raise RuntimeError("Sentence Transformer model is not available. Database cannot function correctly.")

        try:
            # Ensure the directory for the database file exists.
            db_dir = os.path.dirname(db_name)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                logger.info(f"Created database directory: {db_dir}")

            # Establish the database connection.
            self.conn = sqlite3.connect(db_name, check_same_thread=False) # Allow access from different threads (e.g., async tasks)
            # Enable foreign key constraints for data integrity.
            self.conn.execute("PRAGMA foreign_keys = ON")
            # Use Write-Ahead Logging for better concurrency.
            self.conn.execute("PRAGMA journal_mode=WAL")
            # Create a cursor object.
            self.cursor = self.conn.cursor()
            # Set up the database schema if tables don't exist.
            self.setup_tables()
            logger.info(f"Successfully connected to database: {db_name}")
        except sqlite3.Error as e:
            logger.exception(f"Database connection/initialization error for {db_name}: {e}")
            raise # Re-raise the exception to signal failure.
        except RuntimeError as e:
             # This catches the embedding model error from the check above.
             logger.critical(f"Database initialization failed due to missing embedding model: {e}")
             raise

    def close(self):
        """Commits any pending changes and closes the database connection."""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.commit() # Ensure any uncommitted changes are saved.
                self.conn.close()
                logger.info("Database connection closed")
        except sqlite3.Error as e:
            logger.error(f"Error closing database connection: {e}", exc_info=True)

    def execute_query(self, query: str, params: tuple = (), commit: bool = False, fetchone: bool = False, fetchall: bool = False) -> Optional[Union[sqlite3.Cursor, tuple, List[tuple]]]:
        """
        Executes a given SQL query with optional parameters and commit/fetch options.

        Handles basic error logging and rollback on failure.

        Args:
            query: The SQL query string to execute.
            params: A tuple of parameters to substitute into the query (optional).
            commit: Whether to commit the transaction after execution (default: False).
            fetchone: Whether to fetch a single row after execution (default: False).
            fetchall: Whether to fetch all rows after execution (default: False).

        Returns:
            - The sqlite3.Cursor object if no fetch option is specified.
            - A single tuple if fetchone is True.
            - A list of tuples if fetchall is True.
            - None if an error occurs during execution.

        Raises:
            sqlite3.Error: Re-raises SQLite errors after logging and attempting rollback.
        """
        try:
            # Execute the query using the class cursor.
            result = self.cursor.execute(query, params)

            # Commit the transaction if requested.
            if commit:
                self.conn.commit()

            # Fetch results if requested.
            if fetchone:
                return result.fetchone()
            elif fetchall:
                return result.fetchall()
            # Return the cursor object by default.
            return result
        except sqlite3.Error as e:
            # Log the error with query details.
            logger.error(f"Database query error: {e}, Query: {query}, Params: {params}", exc_info=True)
            try:
                # Attempt to rollback the transaction on error.
                self.conn.rollback()
            except sqlite3.Error as rb_err:
                logger.error(f"Rollback failed after query error: {rb_err}", exc_info=True)
            raise # Re-raise the original error.

    def setup_tables(self):
        """
        Creates all necessary database tables and indexes if they don't already exist.

        Defines the schema for users, personas, conversations, stats, embeddings,
        feedback, preferences, and story sessions.
        """
        try:
            logger.debug("Setting up database tables...")
            # --- Users Table ---
            # Stores basic user information.
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,          -- Discord User ID (TEXT for large numbers)
                username TEXT NOT NULL,            -- User's display name (can change)
                created_at TEXT DEFAULT CURRENT_TIMESTAMP, -- Timestamp of first interaction
                last_activity TEXT DEFAULT CURRENT_TIMESTAMP -- Timestamp of last interaction
            )
            ''', commit=True)

            # --- Personas Table ---
            # Stores custom AI personas created by users.
            # The CHECK constraint ensures slot_id is within the allowed range.
            create_personas_sql = f'''
            CREATE TABLE IF NOT EXISTS personas (
                persona_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Internal unique ID for the persona
                user_id TEXT NOT NULL,                     -- Foreign key linking to the user
                slot_id INTEGER NOT NULL,                  -- User-facing slot number (1 to MAX_PERSONAS)
                persona_name TEXT NOT NULL,                -- User-defined name for the persona
                persona_prompt TEXT NOT NULL,              -- The instructions defining the persona's behavior
                created_at TEXT DEFAULT CURRENT_TIMESTAMP, -- Timestamp when the persona was created
                is_active INTEGER DEFAULT 0 CHECK(is_active IN (0, 1)), -- 1 if active, 0 if inactive
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE, -- Delete personas if user is deleted
                UNIQUE (user_id, slot_id)                 -- Each user can only have one persona per slot
                CHECK (slot_id >= 1 AND slot_id <= {self.MAX_PERSONAS}) -- Enforce slot limit
            )'''
            self.execute_query(create_personas_sql, commit=True)
            # Indexes for faster persona lookups.
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_personas_user_slot ON personas (user_id, slot_id);', commit=True)
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_personas_user_active ON personas (user_id, is_active);', commit=True)

            # --- Conversations Table ---
            # Stores individual messages exchanged between users and the bot.
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,      -- Unique ID for each message
                user_id TEXT NOT NULL,                     -- User involved in the conversation
                persona_id INTEGER,                        -- Link to custom persona if applicable (NULL otherwise)
                conversation_type TEXT NOT NULL,           -- Category of interaction (e.g., 'chat', 'persona', 'search')
                role TEXT NOT NULL,                        -- Who sent the message ('User' or bot's role like 'Nemai', 'Sherlock')
                content TEXT NOT NULL,                     -- The actual message text
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,  -- Time the message was recorded
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE, -- Delete messages if user is deleted
                FOREIGN KEY (persona_id) REFERENCES personas (persona_id) ON DELETE SET NULL -- Keep message but remove link if persona is deleted
            )''', commit=True)
            # Indexes for faster history retrieval and filtering.
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_conversations_user_type ON conversations (user_id, conversation_type);', commit=True)
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations (timestamp);', commit=True)
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_conversations_user_role ON conversations (user_id, role);', commit=True)
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_conversations_persona_id ON conversations (persona_id);', commit=True)

            # --- User Stats Table ---
            # Stores usage statistics for each user.
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id TEXT PRIMARY KEY,                  -- Link to the user
                total_messages INTEGER DEFAULT 0,          -- Total user messages across all types
                chat_messages INTEGER DEFAULT 0,           -- Count for /chat
                sherlock_messages INTEGER DEFAULT 0,       -- Count for /sherlock
                teacher_messages INTEGER DEFAULT 0,        -- Count for /teacher
                scientist_messages INTEGER DEFAULT 0,      -- Count for /scientist
                persona_messages INTEGER DEFAULT 0,        -- Count for /persona chat
                doc_assist_messages INTEGER DEFAULT 0,     -- Count for /doc_assist commands
                recommend_messages INTEGER DEFAULT 0,      -- Count for /recommend commands
                story_messages INTEGER DEFAULT 0,          -- Count for /story commands
                last_reset TEXT,                           -- Timestamp of the last history reset
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE -- Delete stats if user is deleted
            )
            ''', commit=True)
            # Ensure new users get a stats entry.
            self.execute_query('''
            INSERT OR IGNORE INTO user_stats (user_id)
            SELECT user_id FROM users WHERE user_id NOT IN (SELECT user_id FROM user_stats)
            ''', commit=True)

            # --- Message Embeddings Table ---
            # Stores vector embeddings for messages to enable semantic search.
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id INTEGER PRIMARY KEY,            -- Link to the conversation message
                embedding BLOB NOT NULL,                   -- The vector embedding stored as a binary blob
                FOREIGN KEY (message_id) REFERENCES conversations (id) ON DELETE CASCADE -- Delete embedding if message is deleted
            )
            ''', commit=True)

            # --- Feedback Table ---
            # Stores user feedback (thumbs up/down) on bot responses.
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique ID for the feedback entry
                user_id TEXT NOT NULL,                     -- User who gave feedback
                message_id INTEGER NOT NULL,               -- The bot message being rated
                user_message_id INTEGER,                   -- The user message that prompted the bot response (optional, for context)
                conversation_type TEXT NOT NULL,           -- Conversation type where feedback occurred
                persona_id INTEGER,                        -- Link to custom persona if applicable
                rating INTEGER NOT NULL CHECK (rating IN (1, -1)), -- 1 for positive, -1 for negative
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,  -- Time feedback was given
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES conversations (id) ON DELETE CASCADE, -- Delete feedback if bot message is deleted
                FOREIGN KEY (user_message_id) REFERENCES conversations (id) ON DELETE SET NULL, -- Keep feedback if user message is deleted
                FOREIGN KEY (persona_id) REFERENCES personas (persona_id) ON DELETE SET NULL, -- Keep feedback if persona is deleted
                UNIQUE (user_id, message_id)              -- Allow only one feedback per user per bot message
            )
            ''', commit=True)
            # Indexes for faster feedback retrieval.
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_feedback_user_type ON feedback (user_id, conversation_type, timestamp);', commit=True)
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_feedback_user_persona ON feedback (user_id, persona_id, timestamp);', commit=True)
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_feedback_message ON feedback (message_id);', commit=True)
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_feedback_user_message ON feedback (user_message_id);', commit=True)

            # --- User Preferences Table ---
            # Stores user preferences for features like recommendations.
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                preference_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique ID for the preference entry
                user_id TEXT NOT NULL,                     -- User these preferences belong to
                preference_type TEXT NOT NULL CHECK(preference_type IN ('movie', 'book', 'music', 'game')), -- Type of preference
                likes TEXT,                                -- Comma-separated list of liked items/genres
                dislikes TEXT,                             -- Comma-separated list of disliked items/genres
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP, -- Timestamp of last update
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                UNIQUE (user_id, preference_type)         -- Only one preference entry per type per user
            )
            ''', commit=True)
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_user_preferences_user_type ON user_preferences (user_id, preference_type);', commit=True)

            # --- Story Sessions Table ---
            # Stores state for ongoing interactive stories.
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS story_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique ID for the story session
                user_id TEXT NOT NULL,                     -- User participating in the story
                genre TEXT,                                -- Optional genre specified by the user
                setting TEXT,                              -- Optional setting specified by the user
                mode TEXT NOT NULL CHECK(mode IN ('collaborative', 'choose_your_own')), -- Story mode
                story_state TEXT,                          -- JSON string storing the story turns/history
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP, -- Timestamp of the last turn
                is_active INTEGER DEFAULT 1,               -- 1 if active, 0 if ended or timed out
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
            ''', commit=True)
            self.execute_query('CREATE INDEX IF NOT EXISTS idx_story_sessions_user_active ON story_sessions (user_id, is_active, last_updated);', commit=True)

            logger.info("Database tables setup or verified complete")
        except sqlite3.Error as e:
            logger.exception(f"Error setting up database tables: {e}")
            raise # Propagate the error.

    def add_user(self, user_id: str, username: str) -> bool:
        """
        Adds a new user to the database or updates their username and last activity time if they already exist.

        Also ensures a corresponding entry exists in the user_stats table.

        Args:
            user_id: The Discord user ID (as a string).
            username: The user's current display name.

        Returns:
            True if the user was successfully added or updated, False otherwise.
        """
        try:
            # Get current time in ISO format (UTC).
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # Use INSERT OR REPLACE (via ON CONFLICT) to add or update user info.
            self.execute_query('''
            INSERT INTO users (user_id, username, created_at, last_activity)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = excluded.username,       -- Update username if it changed
                last_activity = excluded.last_activity -- Always update last activity time
            ''', (user_id, username, now, now), commit=True)

            # Ensure the user has an entry in the stats table, creating one if it doesn't exist.
            self.execute_query('''
            INSERT OR IGNORE INTO user_stats (user_id) VALUES (?)
            ''', (user_id,), commit=True)

            # logger.debug(f"User {user_id} ({username}) added or updated.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding/updating user {user_id}: {e}", exc_info=True)
            return False

    def get_user_stats(self, user_id: str) -> Optional[dict]:
        """
        Retrieves usage statistics for a given user ID.

        Args:
            user_id: The Discord user ID (as a string).

        Returns:
            A dictionary containing the user's statistics (message counts, last reset time),
            or None if the user doesn't exist or an error occurs.
        """
        try:
            # First, check if the user actually exists.
            user_exists = self.execute_query("SELECT 1 FROM users WHERE user_id = ?", (user_id,), fetchone=True)
            if not user_exists:
                logger.warning(f"Attempted to get stats for non-existent user: {user_id}")
                return None # Return None if user not found.

            # Retrieve stats, using LEFT JOIN and COALESCE to handle potential missing stats entry (though add_user should prevent this).
            result = self.execute_query('''
            SELECT
                COALESCE(us.total_messages, 0),
                COALESCE(us.chat_messages, 0),
                COALESCE(us.sherlock_messages, 0),
                COALESCE(us.teacher_messages, 0),
                COALESCE(us.scientist_messages, 0),
                COALESCE(us.persona_messages, 0),
                COALESCE(us.doc_assist_messages, 0),
                COALESCE(us.recommend_messages, 0),
                COALESCE(us.story_messages, 0),
                us.last_reset
            FROM users u
            LEFT JOIN user_stats us ON u.user_id = us.user_id
            WHERE u.user_id = ?
            ''', (user_id,), fetchone=True)

            if result:
                # Format the result into a dictionary.
                return {
                    "total_messages": result[0],
                    "chat_messages": result[1],
                    "sherlock_messages": result[2],
                    "teacher_messages": result[3],
                    "scientist_messages": result[4],
                    "persona_messages": result[5],
                    "doc_assist_messages": result[6],
                    "recommend_messages": result[7],
                    "story_messages": result[8],
                    "last_reset": result[9] # Keep as ISO string or None
                }
            else:
                # This case should be rare due to the initial user check and add_user logic.
                logger.warning(f"No stats found for existing user: {user_id}. Returning defaults.")
                # Return default values if stats row is somehow missing.
                return {
                    "total_messages": 0, "chat_messages": 0, "sherlock_messages": 0,
                    "teacher_messages": 0, "scientist_messages": 0, "persona_messages": 0,
                    "doc_assist_messages": 0, "recommend_messages": 0, "story_messages": 0,
                    "last_reset": None
                }
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user stats for {user_id}: {e}", exc_info=True)
            return None # Return None on database error.

    def save_message(self, user_id: str, conversation_type: str, role: str, content: str, persona_id: Optional[int] = None) -> tuple[Optional[int], Optional[np.ndarray]]:
        """
        Saves a message to the conversations table and its embedding to the message_embeddings table.
        Also updates user statistics if the message is from a user.

        Args:
            user_id: The Discord user ID of the message author (if role is 'User') or recipient.
            conversation_type: The type of conversation (e.g., 'chat', 'persona').
            role: The role of the message sender ('User', 'Nemai', 'Sherlock', etc.).
            content: The text content of the message.
            persona_id: The internal ID of the custom persona used, if applicable (optional).

        Returns:
            A tuple containing:
            - The database ID of the saved message (int) or None if saving failed.
            - The numpy array embedding of the message content (np.ndarray) or None if embedding failed or content was empty.
        """
        # Validate conversation type.
        if conversation_type not in self.VALID_CONVERSATION_TYPES:
            logger.error(f"Invalid conversation_type '{conversation_type}' passed to save_message for user {user_id}.")
            return None, None
        # Ensure embedding model is available.
        if embedding_model is None:
            logger.error("Cannot save message: Embedding model not loaded.")
            return None, None

        embedding = None
        embedding_bytes = None
        # Generate embedding only if content is present.
        if content and content.strip():
            try:
                # Encode content, ensure float32, and convert to bytes for storage.
                embedding = embedding_model.encode(content, convert_to_tensor=False).astype(np.float32)
                embedding_bytes = embedding.tobytes()
            except Exception as model_err:
                logger.error(f"Failed to generate embedding for message content by user {user_id}: {model_err}", exc_info=True)
                embedding = None # Ensure embedding is None if generation fails.
                embedding_bytes = None
        else:
             logger.warning(f"Attempted to save message with empty content for user {user_id}. Skipping embedding.")

        try:
            # Get current timestamp.
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # --- Transaction Start ---
            # Insert into conversations table. Commit is handled later.
            cursor = self.execute_query('''
            INSERT INTO conversations
            (user_id, persona_id, conversation_type, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, persona_id, conversation_type, role, content, now), commit=False)

            # Get the ID of the inserted message.
            message_id = cursor.lastrowid
            if not message_id:
                # This shouldn't happen if the insert succeeded, but check defensively.
                logger.error(f"Failed to get lastrowid after inserting message for user {user_id}.")
                self.conn.rollback() # Rollback the transaction.
                return None, None

            # Insert the embedding if it was generated successfully.
            if embedding_bytes:
                self.execute_query('''
                INSERT INTO message_embeddings (message_id, embedding)
                VALUES (?, ?)
                ''', (message_id, embedding_bytes), commit=False) # Still part of the same transaction.

            # Update user statistics only if it's a user message and a valid type for stats tracking.
            if role.lower() == 'user' and conversation_type in config.VALID_CONVERSATION_TYPES:
                stats_column = f"{conversation_type}_messages"
                try:
                    # Increment total messages and the specific type count.
                    self.execute_query(f'''
                    UPDATE user_stats
                    SET total_messages = total_messages + 1,
                        {stats_column} = {stats_column} + 1
                    WHERE user_id = ?
                    ''', (user_id,), commit=True) # Commit the transaction here after stats update.
                except sqlite3.OperationalError as oe:
                    # Handle potential schema mismatch if a stats column is missing (e.g., after adding a new type).
                    if f"no such column: {stats_column}" in str(oe):
                         logger.error(f"Stats column '{stats_column}' missing during message save for user {user_id}. Stats may be inaccurate. Only updating total_messages.")
                         # Update only total messages as a fallback.
                         self.execute_query('''
                            UPDATE user_stats SET total_messages = total_messages + 1
                            WHERE user_id = ?
                         ''', (user_id,), commit=True) # Commit the transaction.
                    else:
                        # Re-raise other operational errors.
                        self.conn.rollback() # Rollback before raising.
                        raise
            else:
                # If not a user message or not a tracked type, commit the conversation/embedding inserts.
                self.conn.commit()
            # --- Transaction End ---

            # logger.debug(f"Saved message {message_id} for user {user_id}, type {conversation_type}.")
            return message_id, embedding

        except sqlite3.IntegrityError as ie:
             # Handle foreign key or unique constraint violations.
             logger.error(f"Integrity Error saving message for user {user_id}: {ie}", exc_info=True)
             self.conn.rollback()
             if 'FOREIGN KEY constraint failed' in str(ie):
                 # Log specific FK errors for easier debugging.
                 logger.warning(f"Possible missing user or persona during message save for user {user_id}, persona_id {persona_id}")
             return None, None # Return None on integrity errors.
        except sqlite3.Error as e:
            # Handle general SQLite errors.
            logger.error(f"Error saving message for user {user_id}: {e}", exc_info=True)
            self.conn.rollback()
            return None, None
        except Exception as e:
            # Catch any other unexpected errors.
            logger.error(f"Unexpected error saving message for user {user_id}: {e}", exc_info=True)
            self.conn.rollback()
            return None, None

    def get_conversation_history(self, user_id: str, conversation_type: str, limit: int = config.HISTORY_LIMIT_DEFAULT, include_timestamp: bool = False) -> List[str]:
        """
        Retrieves the recent conversation history for a specific user and type.

        Args:
            user_id: The Discord user ID.
            conversation_type: The type of conversation to retrieve (e.g., 'chat', 'persona').
            limit: The maximum number of *turns* (user + bot messages) to retrieve (approximately).
                   Fetches limit*2 messages initially.
            include_timestamp: Whether to prepend a formatted timestamp to each message string.

        Returns:
            A list of strings, where each string represents a message in the format
            "[Timestamp] Role: Content" (if include_timestamp is True) or "Role: Content".
            Returns an empty list if no history is found or an error occurs.
        """
        # Validate conversation type.
        if conversation_type not in self.VALID_CONVERSATION_TYPES:
            logger.warning(f"Invalid conversation type '{conversation_type}' requested for history for user {user_id}")
            return []

        # Fetch roughly twice the limit to account for user/bot turns.
        message_limit = max(1, limit * 2)

        try:
            # Retrieve messages ordered by timestamp descending.
            rows = self.execute_query('''
            SELECT role, content, timestamp FROM conversations
            WHERE user_id = ? AND conversation_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (user_id, conversation_type, message_limit), fetchall=True)

            history = []
            # Reverse the rows to get chronological order and format them.
            for role, content, timestamp_str in reversed(rows):
                if include_timestamp:
                    try:
                        # Format timestamp if requested.
                        dt_obj = datetime.datetime.fromisoformat(timestamp_str)
                        ts_formatted = dt_obj.strftime("%Y-%m-%d %H:%M") # Example format
                        history.append(f"[{ts_formatted}] {role}: {content}")
                    except ValueError:
                         # Fallback if timestamp parsing fails.
                         history.append(f"{role}: {content}")
                else:
                    # Format without timestamp.
                    history.append(f"{role}: {content}")

            # logger.debug(f"Retrieved {len(history)} history messages for user {user_id}, type {conversation_type}.")
            return history
        except sqlite3.Error as e:
            logger.error(f"Error retrieving conversation history for user {user_id}, type {conversation_type}: {e}", exc_info=True)
            return [] # Return empty list on error.

    def get_user_messages(self, user_id: str, conversation_type: Optional[str] = None, limit: int = config.SENTIMENT_LIMIT_DEFAULT) -> List[str]:
        """
        Retrieves the content of recent messages sent *by the user*.

        Used primarily for sentiment analysis where only the user's text is needed.

        Args:
            user_id: The Discord user ID.
            conversation_type: Optional filter for a specific conversation type.
            limit: The maximum number of user messages to retrieve.

        Returns:
            A list of strings, each containing the content of a user message.
            Returns an empty list if no messages are found or an error occurs.
        """
        try:
            # Base SQL query.
            sql = '''
            SELECT content
            FROM conversations
            WHERE user_id = ? AND role = 'User' -- Filter for user messages
            '''
            params = [user_id]

            # Add conversation type filter if provided and valid.
            if conversation_type:
                if conversation_type not in self.VALID_CONVERSATION_TYPES:
                    logger.warning(f"Invalid conversation_type '{conversation_type}' for get_user_messages. Ignoring filter.")
                else:
                    sql += " AND conversation_type = ?"
                    params.append(conversation_type)

            # Add ordering and limit.
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            # Execute and fetch results.
            rows = self.execute_query(sql, tuple(params), fetchall=True)
            # Return only the content (first element of each row tuple).
            return [row[0] for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Error retrieving user messages for user {user_id}, type {conversation_type}: {e}", exc_info=True)
            return []

    def get_relevant_history(self, user_id: str, query_text: str, query_embedding: Optional[np.ndarray], conversation_type: Optional[str] = None,
                             limit: int = config.RELEVANT_HISTORY_LIMIT_DEFAULT,
                             similarity_threshold: float = config.SIMILARITY_THRESHOLD_DEFAULT) -> List[str]:
        """
        Retrieves conversation history relevant to a given query using semantic similarity.

        Combines the most recent messages with messages that are semantically similar
        to the provided query embedding.

        Args:
            user_id: The Discord user ID.
            query_text: The user's current message text (used for logging/context, not embedding).
            query_embedding: The embedding vector of the user's current message.
            conversation_type: Optional filter for a specific conversation type.
            limit: The desired number of relevant history messages to return.
            similarity_threshold: The minimum cosine similarity score for a message to be considered relevant.

        Returns:
            A list of formatted message strings ("Role: Content") deemed relevant,
            sorted chronologically. Returns an empty list on error or if no relevant history found.
        """
        # Prerequisite checks.
        if embedding_model is None:
            logger.error("Cannot get relevant history: Embedding model not loaded.")
            return []
        if query_embedding is None:
            logger.warning(f"No query_embedding provided for relevance search for user {user_id}.")
            # Fallback to just recent history? Or return empty? Returning empty for now.
            return [] # Or potentially call get_conversation_history as fallback

        try:
            # --- Fetch Candidate Messages ---
            # Fetch a larger pool of recent messages than strictly needed, including embeddings.
            # This pool will be filtered for similarity.
            sql = '''
            SELECT c.id, c.role, c.content, c.timestamp, me.embedding
            FROM conversations c
            LEFT JOIN message_embeddings me ON c.id = me.message_id -- Join to get embeddings
            WHERE c.user_id = ?
            '''
            params = [user_id]

            # Add optional conversation type filter.
            if conversation_type:
                if conversation_type not in self.VALID_CONVERSATION_TYPES:
                    logger.warning(f"Invalid conversation_type '{conversation_type}' for relevance search. Ignoring filter.")
                else:
                    sql += " AND c.conversation_type = ?"
                    params.append(conversation_type)

            # Order by most recent and limit the initial fetch size.
            sql += " ORDER BY c.timestamp DESC LIMIT ?"
            params.append(self.RELEVANCE_FETCH_POOL_SIZE)

            fetched_rows = self.execute_query(sql, tuple(params), fetchall=True)

            if not fetched_rows:
                logger.debug(f"No history found for relevance search (User: {user_id}, Type: {conversation_type})")
                return []

            # --- Select Relevant Messages ---
            final_history_candidates = [] # List to store (timestamp, message_string, score) tuples

            # 1. Guarantee Inclusion of Most Recent Messages
            # Always include the N most recent messages regardless of similarity.
            guaranteed_recent = []
            if fetched_rows:
                guaranteed_recent = fetched_rows[:self.RELEVANCE_GUARANTEED_RECENT]
                for msg_id, role, content, timestamp, _ in guaranteed_recent:
                     # Add with a high pseudo-similarity score to ensure they are kept if limit is small.
                     final_history_candidates.append((timestamp, f"{role}: {content}", 1.1))

            # 2. Find Semantically Similar Messages from the Remaining Pool
            remaining_to_consider = []
            if len(fetched_rows) > self.RELEVANCE_GUARANTEED_RECENT:
                remaining_to_consider = fetched_rows[self.RELEVANCE_GUARANTEED_RECENT:]

            similar_messages = []
            if remaining_to_consider and query_embedding is not None:
                # Calculate norm of the query embedding once.
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0: # Avoid division by zero.
                    for msg_id, role, content, timestamp, emb_blob in remaining_to_consider:
                        # Process only if an embedding exists for the message.
                        if emb_blob:
                            try:
                                # Convert blob back to numpy array.
                                stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
                                stored_norm = np.linalg.norm(stored_emb)
                                if stored_norm > 0: # Avoid division by zero.
                                    # Calculate cosine similarity.
                                    similarity = np.dot(query_embedding, stored_emb) / (query_norm * stored_norm)
                                    # Clip similarity score to [-1, 1] range to handle potential float inaccuracies.
                                    similarity = np.clip(similarity, -1.0, 1.0)

                                    # Add if similarity meets the threshold.
                                    if similarity >= similarity_threshold:
                                        similar_messages.append((timestamp, f"{role}: {content}", similarity))
                            except Exception as calc_err:
                                # Log errors during calculation but continue processing other messages.
                                logger.error(f"Error processing embedding/similarity for msg_id {msg_id} (User {user_id}): {calc_err}", exc_info=False)
                                continue # Skip this message on error

            # Sort similar messages by similarity score (highest first).
            similar_messages.sort(key=itemgetter(2), reverse=True)

            # 3. Combine Guaranteed Recent and Top Similar Messages
            # Determine how many more similar messages are needed to reach the limit.
            num_similar_needed = limit - len(final_history_candidates)
            # Create a set of guaranteed messages for efficient duplicate checking.
            guaranteed_content = {msg_str for _, msg_str, _ in final_history_candidates}

            added_similar_count = 0
            for timestamp, msg_str, similarity in similar_messages:
                if added_similar_count >= num_similar_needed:
                    break # Stop if we have enough messages.
                # Add the similar message only if it wasn't already included as a guaranteed recent one.
                if msg_str not in guaranteed_content:
                    final_history_candidates.append((timestamp, msg_str, similarity))
                    added_similar_count += 1

            # --- Final Sorting and Formatting ---
            # Sort the combined list chronologically by timestamp.
            final_history_candidates.sort(key=itemgetter(0))

            # Extract just the message strings for the final result.
            final_history_strings = [msg_str for _, msg_str, _ in final_history_candidates]

            logger.debug(f"Relevant history for user {user_id} (Type: {conversation_type}): Found {len(final_history_strings)} messages ({len(guaranteed_recent)} guaranteed, {added_similar_count} similar).")
            return final_history_strings

        except sqlite3.Error as e:
            logger.error(f"Database error retrieving relevant history for user {user_id}: {e}", exc_info=True)
            return []
        except Exception as e:
            # Catch any other unexpected errors during relevance calculation.
            logger.error(f"Unexpected error during relevance search for user {user_id}: {e}", exc_info=True)
            return []

    def search_user_history(self, user_id: str, query_embedding: np.ndarray, limit: int = config.HISTORY_SEARCH_RESULT_LIMIT, similarity_threshold: float = config.SIMILARITY_THRESHOLD_DEFAULT) -> List[tuple]:
        """
        Performs a semantic search across a user's entire conversation history.

        Finds messages most similar to the query embedding, regardless of conversation type.

        Args:
            user_id: The Discord user ID.
            query_embedding: The embedding vector of the search query.
            limit: The maximum number of search results to return.
            similarity_threshold: The minimum cosine similarity score for a result to be included.

        Returns:
            A list of tuples, where each tuple contains:
            (role, content, timestamp, conversation_type, similarity_score).
            The list is sorted by similarity score (descending).
            Returns an empty list on error or if no results found.
        """
        # Prerequisite checks.
        if embedding_model is None:
            logger.error("Cannot search history: Embedding model not loaded.")
            return []
        if query_embedding is None:
            logger.warning(f"No query_embedding provided for history search for user {user_id}.")
            return []

        try:
            # Fetch a larger pool of messages with embeddings to perform the search on.
            # Fetching more helps ensure we find enough relevant results if history is sparse.
            fetch_limit = limit * 5 # Fetch 5x the desired limit initially.
            sql = '''
            SELECT c.role, c.content, c.timestamp, c.conversation_type, me.embedding
            FROM conversations c
            JOIN message_embeddings me ON c.id = me.message_id -- Only messages with embeddings
            WHERE c.user_id = ?
            ORDER BY c.timestamp DESC -- Start with more recent messages
            LIMIT ?
            '''
            params = (user_id, fetch_limit)
            rows = self.execute_query(sql, params, fetchall=True)

            if not rows:
                return [] # No history with embeddings found for the user.

            # --- Calculate Similarities ---
            similarities = []
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                logger.warning(f"History search query embedding norm is zero for user {user_id}.")
                return [] # Avoid division by zero.

            for role, content, timestamp, conv_type, emb_blob in rows:
                try:
                    # Convert blob to numpy array.
                    stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
                    stored_norm = np.linalg.norm(stored_emb)
                    if stored_norm == 0: continue # Skip invalid embeddings.

                    # Calculate cosine similarity.
                    similarity = np.dot(query_embedding, stored_emb) / (query_norm * stored_norm)
                    similarity = np.clip(similarity, -1.0, 1.0) # Clip for safety.

                    # Add if similarity meets the threshold.
                    if similarity >= similarity_threshold:
                        similarities.append((similarity, role, content, timestamp, conv_type))
                except Exception as calc_err:
                     # Log calculation errors but continue.
                     logger.error(f"Error processing embedding/similarity during history search for user {user_id}: {calc_err}")
                     continue

            # --- Sort and Limit Results ---
            # Sort by similarity score in descending order.
            similarities.sort(reverse=True, key=lambda x: x[0])
            # Return the top 'limit' results in the desired tuple format.
            return [(role, content, timestamp, conv_type, sim) for sim, role, content, timestamp, conv_type in similarities[:limit]]

        except sqlite3.Error as e:
            logger.error(f"Database error searching user history for user {user_id}: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error during user history search for user {user_id}: {e}", exc_info=True)
            return []

    def reset_user_history(self, user_id: str, conversation_type: Optional[str] = None) -> tuple[bool, str]:
        """
        Deletes conversation history for a user, either for a specific type or all types.
        Also resets the corresponding message counts in the user_stats table.

        Args:
            user_id: The Discord user ID.
            conversation_type: The specific conversation type to reset, or None to reset all history.

        Returns:
            A tuple containing:
            - bool: True if the reset was successful, False otherwise.
            - str: A message indicating the outcome of the operation.
        """
        try:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            total_deleted_count = 0
            user_msg_count = 0 # Count of *user* messages deleted (for stats adjustment)

            if conversation_type: # Resetting a specific type
                # Validate the type.
                if conversation_type not in self.VALID_CONVERSATION_TYPES:
                    return False, f"Invalid conversation type '{conversation_type}'. Valid types: {', '.join(self.VALID_CONVERSATION_TYPES)}."

                # Count user messages of this type *before* deleting, for accurate stats update.
                count_result = self.execute_query('''
                SELECT COUNT(*) FROM conversations
                WHERE user_id = ? AND conversation_type = ? AND role = 'User'
                ''', (user_id, conversation_type), fetchone=True)
                user_msg_count = count_result[0] if count_result else 0

                # Count total messages of this type to report deletion count accurately.
                count_result_total = self.execute_query('''
                SELECT COUNT(*) FROM conversations
                WHERE user_id = ? AND conversation_type = ?
                ''', (user_id, conversation_type), fetchone=True)
                total_conv_count = count_result_total[0] if count_result_total else 0

                # If no messages of this type exist, report success immediately.
                if total_conv_count == 0:
                    return True, f"No '{conversation_type}' history found to reset."

                # --- Transaction Start ---
                # Delete the messages.
                self.execute_query('''
                DELETE FROM conversations WHERE user_id = ? AND conversation_type = ?
                ''', (user_id, conversation_type), commit=False) # Commit after stats update.
                total_deleted_count = self.cursor.rowcount

                # Defensive check: ensure deleted count matches expected count.
                if total_deleted_count != total_conv_count:
                     logger.warning(f"Reset history deleted count ({total_deleted_count}) mismatch with initial count ({total_conv_count}) for user {user_id}, type {conversation_type}.")

                # Update the stats table: reset the specific counter and decrement total_messages.
                stats_column = f"{conversation_type}_messages"
                try:
                    self.execute_query(f'''
                    UPDATE user_stats SET
                        total_messages = MAX(0, total_messages - ?), -- Decrement total, ensuring non-negative
                        {stats_column} = 0,                         -- Reset specific type count
                        last_reset = ?                              -- Update last reset time
                    WHERE user_id = ?
                    ''', (user_msg_count, now_iso, user_id), commit=True) # Commit transaction.
                except sqlite3.OperationalError as oe:
                    # Handle missing stats column gracefully.
                    if f"no such column: {stats_column}" in str(oe):
                        logger.error(f"Stats column '{stats_column}' missing during history reset for user {user_id}. Stats may be inaccurate.")
                        self.execute_query('''
                        UPDATE user_stats SET
                            total_messages = MAX(0, total_messages - ?),
                            last_reset = ?
                        WHERE user_id = ?
                        ''', (user_msg_count, now_iso, user_id), commit=True) # Commit transaction.
                    else:
                        self.conn.rollback() # Rollback on other errors.
                        raise
                # --- Transaction End ---

                logger.info(f"Reset {conversation_type} history for user {user_id}, {total_deleted_count} messages deleted (affecting {user_msg_count} user messages in stats).")
                return True, f"Successfully reset your '{conversation_type}' history ({total_deleted_count} messages deleted)."

            else: # Resetting all history
                # Count total messages for reporting.
                count_result_total = self.execute_query('''
                SELECT COUNT(*) FROM conversations WHERE user_id = ?
                ''', (user_id,), fetchone=True)
                total_conv_count = count_result_total[0] if count_result_total else 0

                if total_conv_count == 0:
                    return True, "No history found to reset."

                # --- Transaction Start ---
                # Delete all conversations for the user.
                self.execute_query('''
                DELETE FROM conversations WHERE user_id = ?
                ''', (user_id,), commit=False) # Commit after stats update.
                total_deleted_count = self.cursor.rowcount

                # Reset all message counters in the stats table.
                # Dynamically build the SET clause for all valid types.
                update_cols = ", ".join([f"{ctype}_messages = 0" for ctype in config.VALID_CONVERSATION_TYPES])
                self.execute_query(f'''
                UPDATE user_stats SET
                    total_messages = 0, {update_cols},
                    last_reset = ?
                WHERE user_id = ?
                ''', (now_iso, user_id), commit=True) # Commit transaction.
                # --- Transaction End ---

                logger.info(f"Reset all history for user {user_id}, {total_deleted_count} messages deleted.")
                return True, f"Successfully reset all your history ({total_deleted_count} messages deleted)."

        except sqlite3.Error as e:
            logger.error(f"Error resetting user history for {user_id}, type {conversation_type}: {e}", exc_info=True)
            self.conn.rollback() # Rollback on error.
            return False, f"An error occurred while resetting history. Please try again later."
        except Exception as e:
            # Catch unexpected errors.
            logger.error(f"Unexpected error resetting user history for {user_id}: {e}", exc_info=True)
            self.conn.rollback()
            return False, "An unexpected error occurred while resetting history."

    def prune_old_conversations(self, days: int = config.PRUNE_DAYS) -> int:
        """
        Deletes conversations older than a specified number of days from the database.
        Also attempts to decrement user statistics accordingly.

        Args:
            days: The minimum age in days for a conversation to be pruned.

        Returns:
            The number of messages successfully deleted.
        """
        try:
            # Calculate the cutoff date.
            cutoff_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()
            logger.info(f"Starting pruning of conversations older than {days} days (before {cutoff_date}).")

            # --- Calculate Stats Adjustments (Before Deletion) ---
            # Count the number of *user* messages per type older than the cutoff date.
            stats_to_decrement = self.execute_query('''
            SELECT user_id, conversation_type, COUNT(*) as count
            FROM conversations
            WHERE timestamp < ? AND role = 'User' -- Only count user messages for stats
            GROUP BY user_id, conversation_type
            ''', (cutoff_date,), fetchall=True)

            # --- Delete Old Messages ---
            # Perform the deletion. Commit happens after stats update.
            cursor_delete = self.execute_query('''
            DELETE FROM conversations
            WHERE timestamp < ?
            ''', (cutoff_date,), commit=False)
            deleted_count = cursor_delete.rowcount

            if deleted_count == 0:
                logger.info("No old conversations found to prune.")
                self.conn.rollback() # Nothing to commit.
                return 0

            # Log a warning if messages were deleted but none were user messages (unusual).
            if not stats_to_decrement and deleted_count > 0:
                logger.warning(f"Pruned {deleted_count} messages but found no user messages among them to update stats.")

            # --- Update User Stats ---
            # Decrement stats based on the counts gathered before deletion.
            for user_id, conv_type, count_to_remove in stats_to_decrement:
                if conv_type in self.VALID_CONVERSATION_TYPES:
                    stats_column = f"{conv_type}_messages"
                    try:
                        # Decrement total and specific type counts, ensuring they don't go below zero.
                        self.execute_query(f'''
                            UPDATE user_stats SET
                                total_messages = MAX(0, total_messages - ?),
                                {stats_column} = MAX(0, {stats_column} - ?)
                            WHERE user_id = ?
                            ''',
                            (count_to_remove, count_to_remove, user_id), commit=False) # Part of the main transaction.
                    except sqlite3.OperationalError as oe:
                         # Handle missing stats column gracefully.
                         if f"no such column: {stats_column}" in str(oe):
                              logger.error(f"Stats column '{stats_column}' missing during prune stats update for user {user_id}. Stats may be inaccurate.")
                              # Only decrement total messages as fallback.
                              self.execute_query('''
                                UPDATE user_stats SET
                                    total_messages = MAX(0, total_messages - ?)
                                WHERE user_id = ?
                                ''', (count_to_remove, user_id), commit=False)
                         else:
                              self.conn.rollback() # Rollback on other errors.
                              raise
                else:
                     # Log if an unexpected conversation type is encountered in old messages.
                     logger.warning(f"Encountered unexpected conversation_type '{conv_type}' during prune stats update for user {user_id}.")

            # --- Commit Transaction ---
            self.conn.commit() # Commit deletions and stats updates together.
            logger.info(f"Successfully pruned {deleted_count} messages older than {days} days and updated user stats.")
            return deleted_count

        except sqlite3.Error as e:
            logger.error(f"Error pruning old conversations: {e}", exc_info=True)
            self.conn.rollback() # Rollback on error.
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during pruning: {e}", exc_info=True)
            self.conn.rollback()
            return 0

    # --- Persona Management Methods ---

    def get_persona_count(self, user_id: str) -> int:
        """Counts the number of custom personas created by a user."""
        try:
            result = self.execute_query(
                "SELECT COUNT(*) FROM personas WHERE user_id = ?",
                (user_id,),
                fetchone=True
            )
            return result[0] if result else 0
        except sqlite3.Error as e:
            logger.error(f"Error getting persona count for user {user_id}: {e}", exc_info=True)
            return 0

    def find_available_persona_slot(self, user_id: str) -> Optional[int]:
        """Finds the lowest available persona slot number (1 to MAX_PERSONAS) for a user."""
        try:
            # Get all currently used slot IDs for the user.
            used_slots = self.execute_query(
                "SELECT slot_id FROM personas WHERE user_id = ? ORDER BY slot_id",
                (user_id,),
                fetchall=True
            )
            # Create a set for efficient lookup.
            used_slot_ids = {row[0] for row in used_slots}
            # Iterate through possible slots and return the first one not in the set.
            for slot_id in range(1, self.MAX_PERSONAS + 1):
                if slot_id not in used_slot_ids:
                    return slot_id
            # Return None if all slots are filled.
            return None
        except sqlite3.Error as e:
            logger.error(f"Error finding available persona slot for user {user_id}: {e}", exc_info=True)
            return None

    def add_persona(self, user_id: str, persona_name: str, persona_prompt: str) -> tuple[bool, Union[str, int]]:
        """
        Adds a new custom persona for a user into the next available slot.

        Checks limits, finds an available slot, and ensures the name is unique for the user.

        Args:
            user_id: The Discord user ID.
            persona_name: The desired name for the new persona.
            persona_prompt: The instructions defining the persona's behavior.

        Returns:
            A tuple containing:
            - bool: True if the persona was added successfully, False otherwise.
            - Union[str, int]: The slot ID (int) if successful, or an error message (str) if failed.
        """
        try:
            # Check if the user has reached the maximum persona limit.
            current_count = self.get_persona_count(user_id)
            if current_count >= self.MAX_PERSONAS:
                return False, f"You have reached the maximum limit of {self.MAX_PERSONAS} personas. Please delete an existing one first using `/persona delete`."

            # Find the next available slot number.
            available_slot_id = self.find_available_persona_slot(user_id)
            if available_slot_id is None:
                # This should theoretically not happen if current_count < MAX_PERSONAS, but check defensively.
                logger.warning(f"No available slot found for user {user_id} despite count being {current_count}/{self.MAX_PERSONAS}")
                return False, f"You have reached the maximum limit of {self.MAX_PERSONAS} personas. Please delete one."

            # Check if a persona with the same name (case-insensitive) already exists for this user.
            existing = self.execute_query(
                "SELECT 1 FROM personas WHERE user_id = ? AND lower(persona_name) = lower(?)",
                (user_id, persona_name),
                fetchone=True
            )
            if existing:
                return False, f"A persona named '{persona_name}' already exists. Please choose a different name."

            # Insert the new persona. Default to inactive.
            cursor = self.execute_query(
                '''
                INSERT INTO personas (user_id, slot_id, persona_name, persona_prompt, is_active)
                VALUES (?, ?, ?, ?, 0)
                ''',
                (user_id, available_slot_id, persona_name, persona_prompt),
                commit=True
            )

            # Check if the insert was successful.
            if cursor.rowcount > 0:
                logger.info(f"Persona '{persona_name}' created in slot {available_slot_id} for user {user_id}.")
                return True, available_slot_id # Return success and the assigned slot ID.
            else:
                 # Should not happen if no exception was raised, but log just in case.
                 logger.error(f"Failed to insert persona '{persona_name}' for user {user_id} despite no error.")
                 return False, "Failed to create persona due to an unexpected database issue."

        except sqlite3.IntegrityError as e:
             # Handle specific constraint violations.
             if "UNIQUE constraint failed: personas.user_id, personas.slot_id" in str(e):
                  # This might indicate a race condition if two requests try to grab the same slot simultaneously.
                  logger.error(f"IntegrityError: Attempted to insert duplicate slot {available_slot_id} for user {user_id}. Race condition?", exc_info=True)
                  return False, "An internal error occurred assigning a persona slot. Please try again."
             elif "CHECK constraint failed: personas" in str(e):
                  # This indicates an issue with the slot_id validation (should be caught earlier).
                  logger.error(f"IntegrityError: CHECK constraint failed for slot {available_slot_id}, user {user_id}.", exc_info=True)
                  return False, "Invalid persona slot ID generated. Please report this bug."
             else:
                  # Handle other integrity errors (like potential name uniqueness if added).
                  logger.error(f"Unhandled IntegrityError adding persona '{persona_name}' for user {user_id}: {e}", exc_info=True)
                  return False, f"A persona named '{persona_name}' might already exist, or another constraint was violated."
        except sqlite3.Error as e:
            logger.error(f"Error adding persona '{persona_name}' for user {user_id}: {e}", exc_info=True)
            return False, "An error occurred while adding the persona."

    def get_active_persona(self, user_id: str) -> Optional[tuple]:
        """Retrieves the details of the user's currently active persona, if any."""
        try:
            # Select the persona marked as active (is_active = 1).
            return self.execute_query('''
            SELECT persona_id, slot_id, persona_name, persona_prompt
            FROM personas
            WHERE user_id = ? AND is_active = 1
            ''', (user_id,), fetchone=True)
        except sqlite3.Error as e:
            logger.error(f"Error getting active persona for user {user_id}: {e}", exc_info=True)
            return None

    def get_persona_details_by_slot(self, user_id: str, slot_id: int) -> Optional[tuple]:
         """Retrieves full details of a specific persona by its user-facing slot ID."""
         # Validate slot ID range.
         if not (1 <= slot_id <= self.MAX_PERSONAS):
             logger.warning(f"Invalid slot_id {slot_id} requested for user {user_id}.")
             return None

         try:
             # Fetch all details for the persona in the specified slot.
             return self.execute_query('''
             SELECT persona_id, slot_id, persona_name, persona_prompt, created_at, is_active
             FROM personas
             WHERE user_id = ? AND slot_id = ?
             ''', (user_id, slot_id), fetchone=True)
         except sqlite3.Error as e:
            logger.error(f"Error getting persona details for user {user_id}, slot {slot_id}: {e}", exc_info=True)
            return None

    def get_personas(self, user_id: str) -> List[tuple]:
        """Retrieves a list of all personas created by a user, ordered by slot ID."""
        try:
            # Fetch key details for listing personas.
            return self.execute_query(
                '''
                SELECT slot_id, persona_name, created_at, is_active
                FROM personas
                WHERE user_id = ?
                ORDER BY slot_id ASC
                ''',
                (user_id,),
                fetchall=True
            )
        except sqlite3.Error as e:
            logger.error(f"Error listing personas for user {user_id}: {e}", exc_info=True)
            return [] # Return empty list on error.

    def set_active_persona(self, user_id: str, slot_id_to_activate: int) -> tuple[bool, str]:
        """
        Sets a specific persona as active for the user. Deactivates any previously active persona.

        Args:
            user_id: The Discord user ID.
            slot_id_to_activate: The slot number of the persona to activate.

        Returns:
            A tuple containing:
            - bool: True if activation was successful, False otherwise.
            - str: A message indicating the outcome.
        """
        # Validate slot ID.
        if not (1 <= slot_id_to_activate <= self.MAX_PERSONAS):
            return False, f"Invalid slot ID. Please provide a number between 1 and {self.MAX_PERSONAS}."

        try:
            # --- Check if Persona Exists ---
            # Get the internal ID and name of the persona to activate.
            persona_info = self.execute_query(
                "SELECT persona_id, persona_name FROM personas WHERE user_id = ? AND slot_id = ?",
                (user_id, slot_id_to_activate),
                fetchone=True
            )
            if not persona_info:
                 return False, f"Persona in slot {slot_id_to_activate} not found."

            internal_persona_id_to_activate, persona_name = persona_info

            # --- Transaction Start ---
            # Deactivate any other currently active persona for this user.
            self.execute_query('''
            UPDATE personas SET is_active = 0
            WHERE user_id = ? AND persona_id != ? AND is_active = 1
            ''', (user_id, internal_persona_id_to_activate), commit=False)

            # Activate the target persona.
            result = self.execute_query('''
            UPDATE personas SET is_active = 1
            WHERE user_id = ? AND persona_id = ?
            ''', (user_id, internal_persona_id_to_activate), commit=True) # Commit transaction.
            # --- Transaction End ---

            # Check if the update was successful.
            if result.rowcount > 0:
                logger.info(f"Persona '{persona_name}' in slot {slot_id_to_activate} activated for user {user_id}.")
                return True, f"Persona '{persona_name}' (Slot {slot_id_to_activate}) is now active!"
            else:
                 # This might happen if the persona was already active.
                 logger.warning(f"Activation command for persona slot {slot_id_to_activate} (User {user_id}) affected 0 rows unexpectedly.")
                 # Rollback is not strictly needed here as the commit likely happened, but good practice if logic changes.
                 # self.conn.rollback()
                 # Check if it was already active before returning failure.
                 current_active = self.get_active_persona(user_id)
                 if current_active and current_active[0] == internal_persona_id_to_activate:
                     return True, f"Persona '{persona_name}' (Slot {slot_id_to_activate}) was already active."
                 else:
                     return False, "Failed to activate persona. An unexpected error occurred."

        except sqlite3.Error as e:
            logger.error(f"Error setting active persona slot {slot_id_to_activate} for user {user_id}: {e}", exc_info=True)
            self.conn.rollback() # Rollback on error.
            return False, f"An error occurred while activating the persona."

    def delete_persona(self, user_id: str, slot_id_to_delete: int) -> tuple[bool, str]:
        """
        Deletes a specific custom persona by its slot ID.

        Args:
            user_id: The Discord user ID.
            slot_id_to_delete: The slot number of the persona to delete.

        Returns:
            A tuple containing:
            - bool: True if deletion was successful, False otherwise.
            - str: A message indicating the outcome.
        """
        # Validate slot ID.
        if not (1 <= slot_id_to_delete <= self.MAX_PERSONAS):
            return False, f"Invalid slot ID. Please provide a number between 1 and {self.MAX_PERSONAS}."

        try:
            # --- Check if Persona Exists ---
            # Get details to confirm existence and get the internal ID.
            persona_details = self.get_persona_details_by_slot(user_id, slot_id_to_delete)
            if not persona_details:
                return False, f"Persona in slot {slot_id_to_delete} not found."

            internal_persona_id, _, persona_name, _, _, _ = persona_details

            # --- Delete Persona ---
            # Delete the persona using its internal ID. ON DELETE CASCADE/SET NULL handles FKs.
            cursor = self.execute_query(
                "DELETE FROM personas WHERE user_id = ? AND persona_id = ?",
                (user_id, internal_persona_id),
                commit=True
            )

            # Check if deletion occurred.
            if cursor.rowcount > 0:
                logger.info(f"Persona '{persona_name}' (Slot: {slot_id_to_delete}, Internal ID: {internal_persona_id}) deleted for user {user_id}.")
                return True, f"Persona '{persona_name}' in slot {slot_id_to_delete} deleted successfully."
            else:
                # Should not happen if persona_details were found, but check defensively.
                logger.warning(f"Delete command for persona slot {slot_id_to_delete} (User {user_id}) affected 0 rows unexpectedly.")
                return False, "Failed to delete persona. It might have already been deleted."

        except sqlite3.Error as e:
            logger.error(f"Error deleting persona slot {slot_id_to_delete} for user {user_id}: {e}", exc_info=True)
            self.conn.rollback() # Rollback on error.
            return False, f"An error occurred while deleting the persona."

    # --- Admin & Maintenance Methods ---

    def backup_database(self, backup_dir: str = config.BACKUP_DIR) -> Optional[str]:
        """
        Creates a backup copy of the current SQLite database file.

        Args:
            backup_dir: The directory where the backup file should be saved.

        Returns:
            The full path to the created backup file, or None if backup failed.
        """
        try:
            # Ensure backup directory exists.
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                logger.info(f"Created backup directory: {backup_dir}")

            # Get the path of the currently connected database file.
            db_path_info = self.conn.execute("PRAGMA database_list;").fetchone()
            if not db_path_info or len(db_path_info) < 3:
                 logger.error("Could not determine current database file path via PRAGMA database_list.")
                 return None
            db_name = db_path_info[2] # Path is usually the third element.

            if not db_name or not os.path.exists(db_name):
                 logger.error(f"Database file path '{db_name}' not found or invalid.")
                 return None

            # Create a timestamped backup filename.
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = os.path.join(backup_dir, f"backup_{os.path.basename(db_name)}_{timestamp}.db")

            logger.info(f"Starting database backup to {backup_filename}...")
            # Connect to the backup file.
            backup_conn = sqlite3.connect(backup_filename)
            with backup_conn:
                # Use the online backup API to copy data from the main connection to the backup connection.
                self.conn.backup(backup_conn, pages=0, progress=None) # pages=0 backs up the whole DB.

            backup_conn.close() # Close the backup connection.
            logger.info(f"Database backup completed successfully to {backup_filename}")
            return backup_filename
        except sqlite3.Error as e:
            logger.error(f"Error during database backup: {e}", exc_info=True)
            # Clean up potentially incomplete backup file?
            if 'backup_filename' in locals() and os.path.exists(backup_filename):
                try: os.remove(backup_filename)
                except OSError: pass
            return None
        except Exception as e:
             logger.error(f"Unexpected error during database backup: {e}", exc_info=True)
             return None

    def get_admin_stats(self) -> dict:
        """Retrieves overall statistics about the database for administrative purposes."""
        try:
            # Count entries in various tables.
            user_count = self.execute_query("SELECT COUNT(*) FROM users", fetchone=True)[0]
            message_count = self.execute_query("SELECT COUNT(*) FROM conversations", fetchone=True)[0]
            persona_count = self.execute_query("SELECT COUNT(*) FROM personas", fetchone=True)[0]
            feedback_count = self.execute_query("SELECT COUNT(*) FROM feedback", fetchone=True)[0]
            preference_count = self.execute_query("SELECT COUNT(*) FROM user_preferences", fetchone=True)[0]
            story_count = self.execute_query("SELECT COUNT(*) FROM story_sessions WHERE is_active = 1", fetchone=True)[0]

            # Get the size of the database file.
            db_size_bytes = 0
            try:
                db_path = self.conn.execute("PRAGMA database_list;").fetchone()[2]
                if db_path and os.path.exists(db_path):
                    db_size_bytes = os.path.getsize(db_path)
                elif db_path:
                    logger.warning(f"Database path '{db_path}' reported but not found on disk.")
                else:
                    logger.warning("Could not determine database file path from PRAGMA.")
            except (sqlite3.Error, IndexError, OSError, TypeError) as size_err:
                # Catch various errors that might occur during file size retrieval.
                logger.warning(f"Could not get database file size: {size_err}", exc_info=False)

            # Return stats as a dictionary.
            return {
                "total_users": user_count,
                "total_messages": message_count,
                "total_personas": persona_count,
                "total_feedback": feedback_count,
                "total_preferences": preference_count,
                "active_stories": story_count,
                "database_size_mb": round(db_size_bytes / (1024 * 1024), 2) if db_size_bytes > 0 else 0
            }
        except sqlite3.Error as e:
            logger.error(f"Error retrieving admin stats: {e}", exc_info=True)
            # Return default zero values on error.
            return {
                "total_users": 0, "total_messages": 0, "total_personas": 0,
                "total_feedback": 0, "total_preferences": 0, "active_stories": 0,
                "database_size_mb": 0
            }
        except Exception as e:
            # Catch unexpected errors.
            logger.error(f"Unexpected error in get_admin_stats: {e}", exc_info=True)
            return {
                "total_users": 0, "total_messages": 0, "total_personas": 0,
                "total_feedback": 0, "total_preferences": 0, "active_stories": 0,
                "database_size_mb": 0
            }

    # --- Feedback Methods ---

    def add_feedback(self, user_id: str, bot_message_id: int, user_message_id: Optional[int], conversation_type: str, persona_id: Optional[int], rating: int) -> bool:
        """
        Records user feedback (positive/negative) for a specific bot message.

        Uses INSERT OR REPLACE logic (ON CONFLICT) to allow users to change their feedback.

        Args:
            user_id: The Discord user ID giving feedback.
            bot_message_id: The internal database ID of the bot message being rated.
            user_message_id: The internal database ID of the user message that prompted the bot response.
            conversation_type: The type of conversation where feedback occurred.
            persona_id: The internal ID of the custom persona used, if applicable.
            rating: The feedback rating (1 for positive, -1 for negative).

        Returns:
            True if feedback was successfully recorded or updated, False otherwise.
        """
        # Validate rating and conversation type.
        if rating not in [1, -1]:
            logger.error(f"Invalid rating value '{rating}' provided for bot message {bot_message_id} by user {user_id}.")
            return False
        if conversation_type not in self.VALID_CONVERSATION_TYPES:
            logger.error(f"Invalid conversation_type '{conversation_type}' provided for feedback on bot message {bot_message_id} by user {user_id}.")
            return False

        try:
            # Insert or update feedback. If conflict on (user_id, message_id), update rating and timestamp.
            self.execute_query('''
                INSERT INTO feedback (user_id, message_id, user_message_id, conversation_type, persona_id, rating)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, message_id) DO UPDATE SET
                    rating = excluded.rating,
                    user_message_id = excluded.user_message_id, -- Update user message link too, in case it was missing initially
                    timestamp = CURRENT_TIMESTAMP
            ''', (user_id, bot_message_id, user_message_id, conversation_type, persona_id, rating), commit=True)
            logger.info(f"Feedback ({'positive' if rating == 1 else 'negative'}) recorded for bot message {bot_message_id} (related user msg: {user_message_id}) by user {user_id}.")
            return True
        except sqlite3.IntegrityError as ie:
            # Handle potential foreign key violations if related messages/users don't exist.
            logger.error(f"Integrity Error adding feedback for bot message {bot_message_id} by user {user_id}: {ie}", exc_info=True)
            self.conn.rollback()
            if 'FOREIGN KEY constraint failed' in str(ie):
                logger.warning(f"Failed feedback due to missing FK: user={user_id}, bot_msg={bot_message_id}, user_msg={user_message_id}, persona={persona_id}")
            # The UNIQUE constraint failure is handled by ON CONFLICT, so it shouldn't reach here unless ON CONFLICT fails.
            # elif 'UNIQUE constraint failed: feedback.user_id, feedback.message_id' in str(ie):
            #     logger.warning(f"User {user_id} attempted duplicate feedback for bot message {bot_message_id}. Update handled by ON CONFLICT.")
            #     return True # Technically handled, but log anyway.
            return False
        except sqlite3.Error as e:
            logger.error(f"Error adding feedback for bot message {bot_message_id} by user {user_id}: {e}", exc_info=True)
            self.conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error adding feedback for bot message {bot_message_id} by user {user_id}: {e}", exc_info=True)
            self.conn.rollback()
            return False

    def get_recent_feedback_examples(self, user_id: str, conversation_type: str, persona_id: Optional[int], lookback_days: int, rating: int, limit: int) -> List[tuple[str, str, Optional[bytes]]]:
        """
        Retrieves recent feedback examples (user message, bot response, user message embedding)
        matching specific criteria. Used to provide context to the AI.

        Args:
            user_id: The Discord user ID.
            conversation_type: The type of conversation.
            persona_id: The internal ID of the custom persona (if applicable).
            lookback_days: How many days back to look for feedback.
            rating: The rating to filter by (1 for positive, -1 for negative).
            limit: The maximum number of examples to return.

        Returns:
            A list of tuples, each containing:
            (user_message_content, bot_response_content, user_message_embedding_blob).
            Returns an empty list on error or if no matching examples found.
        """
        # Basic validation.
        if limit <= 0: return []
        if rating not in [1, -1]: return []
        if conversation_type not in self.VALID_CONVERSATION_TYPES: return []

        try:
            # Calculate the cutoff date.
            cutoff_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=lookback_days)).isoformat()

            # SQL query to join feedback with the corresponding user and bot messages, and the user message's embedding.
            sql = '''
            SELECT
                c_user.content AS user_message,        -- User's message content
                c_bot.content AS bot_response,         -- Bot's response content
                me_user.embedding AS user_message_embedding -- Embedding of the user's message
            FROM feedback f
            JOIN conversations c_bot ON f.message_id = c_bot.id -- Join feedback to bot message
            LEFT JOIN conversations c_user ON f.user_message_id = c_user.id -- Join feedback to user message
            LEFT JOIN message_embeddings me_user ON c_user.id = me_user.message_id -- Join user message to its embedding
            WHERE f.user_id = ?                      -- Filter by user
              AND f.conversation_type = ?            -- Filter by conversation type
              AND f.rating = ?                       -- Filter by rating (positive/negative)
              AND f.timestamp >= ?                   -- Filter by time (lookback period)
              AND (? IS NULL OR f.persona_id = ?)    -- Filter by persona ID if provided
              AND c_user.id IS NOT NULL              -- Ensure the user message exists
              AND me_user.embedding IS NOT NULL      -- Ensure the user message has an embedding
            ORDER BY f.timestamp DESC                -- Get the most recent feedback first
            LIMIT ?                                  -- Limit the initial fetch (more than needed for filtering)
            '''

            # Parameters for the query. Note the double persona_id for the IS NULL OR check.
            params = (
                user_id,
                conversation_type,
                rating,
                cutoff_date,
                persona_id, persona_id, # Parameter for IS NULL check and equality check
                limit * 2 # Fetch more initially to allow for filtering/potential missing data
            )

            rows = self.execute_query(sql, params, fetchall=True)

            # Filter out rows where essential data might be missing (though query constraints should prevent this).
            valid_rows = [row for row in rows if row[0] is not None and row[2] is not None] # Check user message content and embedding
            # Return only the requested number of valid rows.
            return valid_rows[:limit] if valid_rows else []

        except sqlite3.Error as e:
            logger.error(f"Error retrieving feedback examples for user {user_id}, type {conversation_type}, persona {persona_id}, rating {rating}: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error retrieving feedback examples: {e}", exc_info=True)
            return []

    # --- Context Retrieval for Sentiment Analysis ---

    def get_user_message_contexts(self, user_id: str, conversation_type: Optional[str] = None, limit: int = config.SENTIMENT_LIMIT_DEFAULT) -> List[Dict[str, str]]:
            """
            Retrieves recent user messages along with the immediately preceding bot message context.

            Used for contextual sentiment analysis where the bot's prior message might influence
            the interpretation of the user's sentiment.

            Args:
                user_id: The Discord user ID.
                conversation_type: Optional filter for a specific conversation type.
                limit: The maximum number of user message contexts to retrieve.

            Returns:
                A list of dictionaries, each containing:
                {'user_message': str, 'bot_context': str}.
                Returns an empty list on error or if no contexts found.
            """
            if limit <= 0:
                return []
            try:
                # Fetch a larger chunk of recent history for the user/type.
                base_sql = '''
                SELECT role, content
                FROM conversations
                WHERE user_id = ?
                '''
                params = [user_id]

                if conversation_type:
                    if conversation_type not in self.VALID_CONVERSATION_TYPES:
                        logger.warning(f"Invalid conversation_type '{conversation_type}' for get_user_message_contexts. Ignoring filter.")
                    else:
                        base_sql += " AND conversation_type = ?"
                        params.append(conversation_type)

                # Fetch more messages than needed (e.g., 4x limit) to increase chances of finding pairs.
                fetch_limit = limit * 4
                base_sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(fetch_limit)

                rows = self.execute_query(base_sql, tuple(params), fetchall=True)
                if not rows:
                    return [] # No history found.

                # Reverse rows to process in chronological order.
                rows.reverse()

                contexts = []
                # Process messages to find user messages preceded by a bot message.
                # Keep a small buffer of recent messages.
                message_buffer = []
                max_buffer_size = 5 # How many messages back to look for context.

                for role, content in rows:
                    current_message = {'role': role.lower(), 'content': content}

                    # If the current message is from the user...
                    if current_message['role'] == 'user':
                        preceding_bot_message_content = None
                        # Look backwards in the buffer for the most recent non-user message.
                        for i in range(len(message_buffer) - 1, -1, -1):
                            if message_buffer[i]['role'] != 'user':
                                preceding_bot_message_content = message_buffer[i]['content']
                                break # Found the most recent bot context.

                        # If bot context was found, add the pair to the results.
                        if preceding_bot_message_content:
                            contexts.append({
                                'user_message': current_message['content'],
                                'bot_context': preceding_bot_message_content
                            })
                            # Stop if we've reached the desired limit.
                            if len(contexts) >= limit:
                                break

                    # Add the current message to the buffer and maintain buffer size.
                    message_buffer.append(current_message)
                    if len(message_buffer) > max_buffer_size:
                        message_buffer.pop(0) # Remove the oldest message.

                # Return the found contexts (up to the limit).
                return contexts # Already limited within the loop.

            except sqlite3.Error as e:
                logger.error(f"Error retrieving user message contexts for user {user_id}, type {conversation_type}: {e}", exc_info=True)
                return []
            except Exception as e:
                logger.error(f"Unexpected error retrieving user message contexts for user {user_id}: {e}", exc_info=True)
                return []

    # --- History Export Method ---

    def get_conversation_history_for_export(self, user_id: str, conversation_type: Optional[str] = None, limit: int = config.EXPORT_LIMIT_DEFAULT) -> List[tuple]:
        """
        Retrieves conversation history formatted for export to a file.

        Args:
            user_id: The Discord user ID.
            conversation_type: Optional filter for a specific conversation type.
            limit: The maximum number of messages to retrieve.

        Returns:
            A list of tuples, each containing (role, content, timestamp_str),
            ordered chronologically (oldest first). Returns empty list on error.
        """
        if limit <= 0:
            return []

        try:
            sql = '''
            SELECT role, content, timestamp
            FROM conversations
            WHERE user_id = ?
            '''
            params = [user_id]

            # Add optional type filter.
            if conversation_type:
                if conversation_type not in self.VALID_CONVERSATION_TYPES:
                    logger.warning(f"Invalid conversation_type '{conversation_type}' for export. Ignoring filter.")
                else:
                    sql += " AND conversation_type = ?"
                    params.append(conversation_type)

            # Order chronologically and apply limit.
            sql += " ORDER BY timestamp ASC LIMIT ?" # ASC for chronological export
            params.append(limit)

            rows = self.execute_query(sql, tuple(params), fetchall=True)
            return rows if rows else []

        except sqlite3.Error as e:
            logger.error(f"Error retrieving conversation history for export for user {user_id}, type {conversation_type}: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error retrieving history for export user {user_id}: {e}", exc_info=True)
            return []

    # --- User Preference Methods ---

    def set_user_preference(self, user_id: str, preference_type: str, likes: Optional[str] = None, dislikes: Optional[str] = None) -> tuple[bool, str]:
        """
        Sets or updates user preferences (likes/dislikes) for a specific category (e.g., 'movie').

        Args:
            user_id: The Discord user ID.
            preference_type: The category of preference ('movie', 'book', 'music', 'game').
            likes: A comma-separated string of liked items/genres (optional).
            dislikes: A comma-separated string of disliked items/genres (optional).

        Returns:
            A tuple containing:
            - bool: True if preferences were saved successfully, False otherwise.
            - str: A message indicating the outcome.
        """
        valid_types = ['movie', 'book', 'music', 'game']
        if preference_type not in valid_types:
            logger.error(f"Invalid preference type '{preference_type}' for user {user_id}")
            return False, f"Invalid preference type. Choose from: {', '.join(valid_types)}"

        # Clean up input strings (strip whitespace around commas).
        likes_str = ", ".join([s.strip() for s in likes.split(',') if s.strip()]) if likes else None
        dislikes_str = ", ".join([s.strip() for s in dislikes.split(',') if s.strip()]) if dislikes else None

        # Validate length limits from config.
        if likes_str and len(likes_str) > config.MAX_USER_PREFERENCES:
            return False, f"Likes list is too long (max {config.MAX_USER_PREFERENCES} characters)."
        if dislikes_str and len(dislikes_str) > config.MAX_USER_PREFERENCES:
            return False, f"Dislikes list is too long (max {config.MAX_USER_PREFERENCES} characters)."

        try:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            # Use INSERT OR REPLACE (via ON CONFLICT) to add or update preferences.
            self.execute_query('''
                INSERT INTO user_preferences (user_id, preference_type, likes, dislikes, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, preference_type) DO UPDATE SET
                    likes = excluded.likes,
                    dislikes = excluded.dislikes,
                    last_updated = excluded.last_updated
            ''', (user_id, preference_type, likes_str, dislikes_str, now), commit=True)
            logger.info(f"Preferences for '{preference_type}' updated for user {user_id}.")
            return True, f"Preferences for {preference_type} updated successfully."
        except sqlite3.Error as e:
            logger.error(f"Error setting preferences for user {user_id}, type {preference_type}: {e}", exc_info=True)
            return False, "An error occurred while saving your preferences."

    def get_user_preferences(self, user_id: str, preference_type: str) -> Optional[tuple[Optional[str], Optional[str]]]:
        """
        Retrieves the stored likes and dislikes for a user and preference type.

        Args:
            user_id: The Discord user ID.
            preference_type: The category of preference ('movie', 'book', 'music', 'game').

        Returns:
            A tuple containing (likes, dislikes) as strings, or (None, None) if no preferences
            are set for that type or an error occurs.
        """
        valid_types = ['movie', 'book', 'music', 'game']
        if preference_type not in valid_types:
            return None # Return None for invalid type.

        try:
            result = self.execute_query('''
                SELECT likes, dislikes FROM user_preferences
                WHERE user_id = ? AND preference_type = ?
            ''', (user_id, preference_type), fetchone=True)
            # Return the result tuple (likes, dislikes) or (None, None) if no row found.
            return result if result else (None, None)
        except sqlite3.Error as e:
            logger.error(f"Error retrieving preferences for user {user_id}, type {preference_type}: {e}", exc_info=True)
            return None # Return None on error.

    # --- Story Session Methods ---

    def create_story_session(self, user_id: str, genre: Optional[str], setting: Optional[str], mode: str) -> tuple[Optional[int], Optional[str]]:
        """
        Creates a new interactive story session for a user.

        Args:
            user_id: The Discord user ID starting the story.
            genre: The optional genre for the story.
            setting: The optional setting description.
            mode: The story mode ('collaborative' or 'choose_your_own').

        Returns:
            A tuple containing:
            - int | None: The unique ID of the created session, or None on failure.
            - str | None: An error message if creation failed, None otherwise.
        """
        # Validate mode.
        if mode not in ['collaborative', 'choose_your_own']:
            return None, "Invalid story mode selected."

        try:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            # Initialize story state as an empty JSON list.
            initial_state = json.dumps([])
            # Insert the new session record.
            cursor = self.execute_query('''
                INSERT INTO story_sessions (user_id, genre, setting, mode, story_state, last_updated, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1) -- Start as active
            ''', (user_id, genre, setting, mode, initial_state, now), commit=True)
            session_id = cursor.lastrowid
            logger.info(f"Created new '{mode}' story session {session_id} for user {user_id} (Genre: {genre}).")
            return session_id, None # Return session ID and no error message.
        except sqlite3.Error as e:
            logger.error(f"Error creating story session for user {user_id}: {e}", exc_info=True)
            return None, "An error occurred while starting the story session." # Return None ID and error message.

    def get_active_story_session(self, user_id: str) -> Optional[tuple]:
        """
        Retrieves the most recent active story session for a user that hasn't timed out.

        Also automatically marks timed-out sessions as inactive.

        Args:
            user_id: The Discord user ID.

        Returns:
            A tuple containing session details (session_id, genre, setting, mode, story_state, last_updated)
            if an active, non-timed-out session exists, otherwise None.
        """
        try:
            # Calculate the timeout limit based on config.
            timeout_limit = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=config.STORY_TIMEOUT_MINUTES)).isoformat()

            # Query for the most recent active session updated within the timeout period.
            result = self.execute_query('''
                SELECT session_id, genre, setting, mode, story_state, last_updated
                FROM story_sessions
                WHERE user_id = ? AND is_active = 1 AND last_updated >= ?
                ORDER BY last_updated DESC
                LIMIT 1
            ''', (user_id, timeout_limit), fetchone=True)

            # If no active session found within the timeout...
            if not result:
                # Mark any older active sessions for this user as inactive.
                self.execute_query('''
                    UPDATE story_sessions SET is_active = 0
                    WHERE user_id = ? AND is_active = 1 AND last_updated < ?
                ''', (user_id, timeout_limit), commit=True)
                logger.debug(f"No active (non-timed-out) story session found for user {user_id}.")
                return None # No active session found.

            # Return the details of the active session.
            return result
        except sqlite3.Error as e:
            logger.error(f"Error getting active story session for user {user_id}: {e}", exc_info=True)
            return None

    def get_story_session_details(self, session_id: int, user_id: str) -> Optional[tuple]:
        """
        Gets details for a specific, active story session owned by the user.
        Used primarily to validate session ownership before processing choices.

        Args:
            session_id: The ID of the story session to retrieve.
            user_id: The ID of the user expected to own the session.

        Returns:
            A tuple containing session details if found, active, and owned by the user, otherwise None.
        """
        try:
            result = self.execute_query('''
                SELECT session_id, genre, setting, mode, story_state, last_updated
                FROM story_sessions
                WHERE session_id = ? AND user_id = ? AND is_active = 1
            ''', (session_id, user_id), fetchone=True)
            return result # Returns the row tuple or None if not found/inactive/wrong user.
        except sqlite3.Error as e:
            logger.error(f"Error getting details for story session {session_id}, user {user_id}: {e}", exc_info=True)
            return None

    def update_story_session(self, session_id: int, new_turn: dict) -> tuple[bool, Optional[str]]:
        """
        Appends a new turn (user input or AI response) to the story session's state.

        Args:
            session_id: The ID of the story session to update.
            new_turn: A dictionary representing the turn (e.g., {'role': 'User', 'content': '...'}).

        Returns:
            A tuple containing:
            - bool: True if the update was successful, False otherwise.
            - str | None: An error message if the update failed, None otherwise.
        """
        try:
            # --- Get Current State ---
            # Retrieve the current story state JSON.
            current_state_row = self.execute_query(
                "SELECT story_state FROM story_sessions WHERE session_id = ? AND is_active = 1",
                (session_id,),
                fetchone=True
            )
            # Check if the session exists and is active.
            if not current_state_row:
                logger.warning(f"Attempted to update inactive or non-existent story session {session_id}")
                return False, "Story session not found or is inactive."

            # --- Update State ---
            # Load the JSON history, append the new turn.
            story_history = json.loads(current_state_row[0])
            story_history.append(new_turn)

            # Keep only the last N turns to manage context length (defined in config).
            story_history = story_history[-config.STORY_CONTEXT_LIMIT:]

            # Convert back to JSON string.
            updated_state_json = json.dumps(story_history)
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # --- Save Updated State ---
            # Update the database record with the new state and timestamp.
            self.execute_query('''
                UPDATE story_sessions
                SET story_state = ?, last_updated = ?
                WHERE session_id = ?
            ''', (updated_state_json, now, session_id), commit=True)
            # logger.debug(f"Updated story session {session_id}.")
            return True, None # Success, no error message.
        except json.JSONDecodeError as je:
             # Handle errors decoding the existing story state.
             logger.error(f"Error decoding JSON story state for session {session_id}: {je}", exc_info=True)
             # Consider ending the session here as it's corrupted. db.end_story_session(session_id)
             return False, "Error reading story history."
        except sqlite3.Error as e:
            logger.error(f"Error updating story session {session_id}: {e}", exc_info=True)
            return False, "An error occurred while updating the story."
        except Exception as e:
             # Catch any other unexpected errors.
             logger.error(f"Unexpected error updating story session {session_id}: {e}", exc_info=True)
             return False, "An unexpected error occurred while updating the story."

    def end_story_session(self, session_id: int) -> bool:
        """
        Marks a story session as inactive.

        Args:
            session_id: The ID of the story session to end.

        Returns:
            True if the session was marked inactive, False otherwise (e.g., session not found).
        """
        try:
            # Set the is_active flag to 0.
            result = self.execute_query(
                "UPDATE story_sessions SET is_active = 0 WHERE session_id = ?",
                (session_id,),
                commit=True
            )
            if result.rowcount > 0:
                logger.info(f"Ended story session {session_id}.")
                return True
            else:
                logger.warning(f"Attempted to end session {session_id}, but it was not found or already inactive.")
                return False # Return False if no rows were affected.
        except sqlite3.Error as e:
            logger.error(f"Error ending story session {session_id}: {e}", exc_info=True)
            return False

# END OF FILE database.py
