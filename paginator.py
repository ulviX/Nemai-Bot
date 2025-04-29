# START OF FILE paginator.py

"""
Paginator Module for Discord Bot Responses

Provides functionality to split long messages into multiple pages and display them
using interactive buttons in Discord embeds.

Functions:
    chunk_message: Splits a long string into smaller chunks suitable for embed descriptions.

Classes:
    PaginatorView: A discord.ui.View subclass that manages pagination buttons and embed updates.
"""

import discord
import config # For configuration constants like MAX_PAGE_SIZE, EMBED_COLORS
import logging
from typing import List, Union, Any # Using Any for broader input type hint in chunk_message initially

logger = logging.getLogger('paginator') # Logger specific to this module

def chunk_message(text: Union[str, Any], chunk_size: int = config.MAX_PAGE_SIZE) -> List[str]:
    """
    Splits a long string into chunks smaller than a specified size.

    Attempts to split intelligently by paragraphs (`\n\n`), then sentences ('. '),
    and finally performs hard breaks if necessary. Aims to preserve readability.

    Args:
        text: The input text to chunk. Can be any type that can be converted to a string.
        chunk_size: The maximum character size for each chunk. Defaults to `config.MAX_PAGE_SIZE`.

    Returns:
        A list of strings, where each string is a chunk of the original text.
        Returns an empty list if the input cannot be converted to a string or is empty.
        Returns a list with the original text if it's already within the chunk size.
    """
    # --- Input Validation and Conversion ---
    # Ensure the input is a string.
    if not isinstance(text, str):
        try:
            text = str(text) # Attempt conversion
        except Exception:
            logger.error("Failed to convert input to string for chunking.")
            return [] # Return empty list if conversion fails

    # Handle empty input string.
    if not text:
        return []

    # If the text is already short enough, return it as a single chunk.
    if len(text) <= chunk_size:
        return [text]

    # --- Chunking Logic ---
    chunks = [] # List to store the resulting chunks
    current_chunk = "" # Accumulator for the current chunk being built

    # Split the text into paragraphs first, preserving double newlines.
    paragraphs = text.split('\n\n')

    for i, paragraph in enumerate(paragraphs):
        # Re-add the double newline separator, except for the last paragraph.
        paragraph_with_separator = paragraph + ('\n\n' if i < len(paragraphs) - 1 else '')

        # Handle empty paragraphs (just double newlines).
        if not paragraph.strip() and paragraph_with_separator == '\n\n':
            # If adding the empty paragraph fits, add it.
            if len(current_chunk) + len(paragraph_with_separator) <= chunk_size:
                current_chunk += paragraph_with_separator
            else:
                # Otherwise, finalize the current chunk and start a new one with the empty paragraph (if it fits).
                if current_chunk:
                    chunks.append(current_chunk)
                # Only start new chunk if the separator itself fits.
                current_chunk = paragraph_with_separator if len(paragraph_with_separator) <= chunk_size else ""
            continue # Move to the next paragraph

        # --- Handle Paragraphs Longer Than chunk_size ---
        if len(paragraph_with_separator) > chunk_size:
            # If the current paragraph itself is too long, finalize the previous chunk.
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Split the oversized paragraph further. Try splitting by sentences first.
            # Note: This sentence splitting is basic and might not handle all edge cases perfectly (e.g., abbreviations).
            sentences = paragraph_with_separator.split('. ')
            temp_chunk = "" # Accumulator for sentence parts within the large paragraph
            for sentence_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence: continue # Skip empty strings resulting from split

                # Re-add the period and space, except possibly for the very last part.
                # Check if it's the last part of the split *and* the last part of the original paragraph.
                is_last_sentence_part = (sentence_idx == len(sentences) - 1)
                original_ends_with_period = paragraph_with_separator.endswith('.')
                if not (is_last_sentence_part and not original_ends_with_period):
                    sentence += ". "

                # If adding the sentence fits in the temp_chunk...
                if len(temp_chunk) + len(sentence) <= chunk_size:
                    temp_chunk += sentence
                else:
                    # Finalize the previous temp_chunk.
                    if temp_chunk: chunks.append(temp_chunk)
                    # If the sentence *itself* is too long, perform hard breaks.
                    if len(sentence) > chunk_size:
                        for k in range(0, len(sentence), chunk_size):
                            chunks.append(sentence[k:k+chunk_size])
                        temp_chunk = "" # Reset temp_chunk after hard break
                    else:
                        # Start a new temp_chunk with the current sentence.
                        temp_chunk = sentence
            # Add any remaining part in temp_chunk.
            if temp_chunk: chunks.append(temp_chunk)

        # --- Handle Paragraphs Shorter Than chunk_size ---
        # If adding the current paragraph exceeds the chunk size...
        elif len(current_chunk) + len(paragraph_with_separator) > chunk_size:
            # Finalize the previous chunk.
            if current_chunk:
                chunks.append(current_chunk)
            # Start a new chunk with the current paragraph.
            current_chunk = paragraph_with_separator
        # If the current paragraph fits...
        else:
            # Add it to the current chunk.
            current_chunk += paragraph_with_separator

    # Add the last accumulated chunk if it contains text.
    if current_chunk:
        chunks.append(current_chunk)

    # Filter out any potentially empty chunks that might have been created.
    final_chunks = [chunk for chunk in chunks if chunk.strip()]

    # Fallback: If chunking somehow resulted in empty list, return the truncated original text.
    return final_chunks if final_chunks else [text[:chunk_size]]


class PaginatorView(discord.ui.View):
    """
    A Discord UI View for paginating long text content within embeds.

    Provides buttons (First, Previous, Next, Last, Close) to navigate through pages.
    Automatically disables buttons when they are not applicable (e.g., Previous on page 1).

    Attributes:
        pages (List[str]): The list of text chunks (pages) to display.
        current_page (int): The index of the currently displayed page.
        author (Union[discord.User, discord.Member]): The user who initiated the command, used for interaction checks.
        title (str): The base title for the embed. Page numbers are appended automatically.
        color_key (str): The key to look up the embed color in `config.EMBED_COLORS`.
        message (Optional[discord.Message]): The message object this view is attached to. Set after sending.
    """
    def __init__(self, pages: List[str], author: Union[discord.User, discord.Member], title: str = "", color_key: str = "default", timeout: float = config.PAGINATOR_TIMEOUT):
        """
        Initializes the PaginatorView.

        Args:
            pages: A list of strings, where each string is a page of content.
            author: The user who triggered the command.
            title: The base title for the embed.
            color_key: The key for the embed color in `config.EMBED_COLORS`.
            timeout: The duration (in seconds) after which the view becomes inactive.
        """
        super().__init__(timeout=timeout)
        # Ensure pages list is not empty; provide a placeholder if it is.
        if not pages:
             logger.error("PaginatorView initiated with zero pages. Creating a placeholder.")
             pages = ["Error: No content to display."] # Prevent errors later

        self.pages = pages
        self.current_page = 0
        self.author = author
        self.title = title
        self.color_key = color_key
        self.message: Optional[discord.Message] = None # Will be set after the message is sent

        # Set initial button states (e.g., disable Prev/First on page 0).
        self._update_button_states()

    def _update_button_states(self):
        """
        Updates the enabled/disabled state of the navigation buttons based on the current page.
        """
        page_count = len(self.pages)
        is_first_page = self.current_page == 0
        is_last_page = self.current_page == page_count - 1

        # Iterate through the view's children (buttons).
        for item in self.children:
            if not isinstance(item, discord.ui.Button): continue # Skip non-buttons

            # Disable buttons based on current page index.
            if item.custom_id == "page_first": item.disabled = is_first_page
            elif item.custom_id == "page_prev": item.disabled = is_first_page
            elif item.custom_id == "page_next": item.disabled = is_last_page
            elif item.custom_id == "page_last": item.disabled = is_last_page

        # If there's only one page, disable all navigation buttons except Close.
        if page_count <= 1:
            for item in self.children:
                 if isinstance(item, discord.ui.Button) and item.custom_id.startswith("page_"):
                     if item.custom_id != "page_close": # Keep close button enabled
                         item.disabled = True

    def get_page_embed(self) -> discord.Embed:
        """
        Creates the Discord Embed object for the current page.

        Handles potential index errors and content truncation if a page
        somehow exceeds the Discord embed description limit.

        Returns:
            A discord.Embed object representing the current page.
        """
        # Get the content for the current page.
        try:
            page_content = self.pages[self.current_page]
        except IndexError:
             # Handle rare case where current_page might be out of bounds.
             logger.warning(f"Current page index {self.current_page} out of bounds for pages list (len {len(self.pages)}). Resetting to 0.")
             self.current_page = 0
             page_content = self.pages[self.current_page] # Get content for page 0

        # Truncate content if it exceeds Discord's limit (should ideally be handled by chunk_message).
        if len(page_content) > config.MAX_EMBED_DESCRIPTION:
            logger.warning(f"Page content exceeds MAX_EMBED_DESCRIPTION ({config.MAX_EMBED_DESCRIPTION}). Truncating.")
            page_content = page_content[:config.MAX_EMBED_DESCRIPTION - 4] + " ..."

        # Format the embed title with page numbers if there's more than one page.
        page_title = self.title
        if len(self.pages) > 1:
             page_title = f"{self.title} (Page {self.current_page + 1}/{len(self.pages)})"

        # Get the embed color from config, defaulting if key not found or invalid.
        embed_color = config.EMBED_COLORS.get(self.color_key, discord.Color.blue())
        if not isinstance(embed_color, discord.Color):
             logger.warning(f"Invalid color found for key '{self.color_key}', defaulting to blue.")
             embed_color = discord.Color.blue()

        # Create the embed object.
        embed = discord.Embed(
            title=page_title,
            description=page_content,
            color=embed_color
        )

        # Set the footer with the requesting user's name and avatar.
        if self.author:
            try:
                icon_url = self.author.display_avatar.url
            except AttributeError: # Handle cases where avatar might not be available
                icon_url = None
            embed.set_footer(text=f"Requested by: {self.author.display_name}", icon_url=icon_url)

        return embed

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """
        Checks if the user interacting with the buttons is the original author.

        Prevents other users from controlling someone else's paginator.

        Args:
            interaction: The interaction object from the button click.

        Returns:
            True if the interacting user is the author, False otherwise.
        """
        # Compare the interacting user's ID with the stored author's ID.
        if interaction.user.id != self.author.id:
            # Send an ephemeral message if the user is not authorized.
            await interaction.response.send_message("These buttons aren’t for you.", ephemeral=True)
            return False
        return True # Allow the interaction.

    async def update_message(self, interaction: discord.Interaction):
        """
        Updates the message with the new page embed and adjusted button states.

        Called after a navigation button is successfully clicked.

        Args:
            interaction: The interaction object from the button click.
        """
        # Update button enabled/disabled states first.
        self._update_button_states()
        # Edit the original message with the new embed and updated view (buttons).
        await interaction.response.edit_message(embed=self.get_page_embed(), view=self)

    # --- Button Callbacks ---
    # Each button uses the @discord.ui.button decorator.
    # The custom_id is used internally by Discord and in _update_button_states.

    @discord.ui.button(label="⏮️", style=discord.ButtonStyle.primary, custom_id="page_first")
    async def first(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Callback for the 'First Page' button."""
        # Check if already on the first page to avoid unnecessary updates.
        if self.current_page != 0:
            self.current_page = 0
            await self.update_message(interaction)
        else:
            # If already on the first page, just defer the interaction to acknowledge the click.
            await interaction.response.defer()

    @discord.ui.button(label="◀️", style=discord.ButtonStyle.primary, custom_id="page_prev")
    async def previous(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Callback for the 'Previous Page' button."""
        if self.current_page > 0:
            self.current_page -= 1
            await self.update_message(interaction)
        else:
            await interaction.response.defer() # Already on first page

    @discord.ui.button(label="▶️", style=discord.ButtonStyle.primary, custom_id="page_next")
    async def next_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Callback for the 'Next Page' button."""
        if self.current_page < len(self.pages) - 1:
            self.current_page += 1
            await self.update_message(interaction)
        else:
             await interaction.response.defer() # Already on last page

    @discord.ui.button(label="⏭️", style=discord.ButtonStyle.primary, custom_id="page_last")
    async def last(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Callback for the 'Last Page' button."""
        if self.current_page != len(self.pages) - 1:
            self.current_page = len(self.pages) - 1
            await self.update_message(interaction)
        else:
            await interaction.response.defer() # Already on last page

    @discord.ui.button(label="❌", style=discord.ButtonStyle.danger, custom_id="page_close")
    async def close(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Callback for the 'Close Paginator' button."""
        # Attempt to delete the message containing the paginator.
        if interaction.message:
            try:
                await interaction.message.delete()
            except discord.NotFound:
                 # Message might have been deleted manually already.
                 logger.debug(f"Message {interaction.message.id} already deleted when closing paginator.")
            except discord.Forbidden:
                 # Bot lacks permissions to delete the message.
                 logger.warning(f"Missing permissions to delete message {interaction.message.id} on paginator close.")
            except discord.HTTPException as e:
                 # Other Discord API errors.
                 logger.error(f"HTTP error deleting message {interaction.message.id} on paginator close: {e}")
            except Exception as e:
                 # Catch any other unexpected errors during deletion.
                 logger.exception(f"Unexpected error deleting message {interaction.message.id} on paginator close: {e}")

        # Defer the interaction response if the message deletion failed or didn't happen,
        # otherwise the interaction might show as "failed".
        try:
            await interaction.response.defer()
        except discord.NotFound: # Interaction might already be gone
            pass
        # Stop the view from listening for further interactions.
        self.stop()

    async def on_timeout(self):
        """
        Called when the view times out (no interaction within the specified timeout period).

        Disables all buttons on the message if it still exists and is not ephemeral.
        """
        if self.message:
            # Check if the message is ephemeral (cannot be edited after timeout).
            # Note: This flag check might not be perfectly reliable in all discord.py versions or scenarios.
            is_ephemeral = (self.message.flags.value >> 6) & 1

            if not is_ephemeral:
                try:
                    # Check if buttons are already disabled (e.g., by the close button).
                    already_disabled = all(item.disabled for item in self.children if isinstance(item, discord.ui.Button))
                    if not already_disabled:
                        # Disable all buttons.
                        for item in self.children:
                            if isinstance(item, discord.ui.Button):
                                item.disabled = True
                        # Edit the message to show the disabled buttons.
                        await self.message.edit(view=self)
                except discord.NotFound:
                    # Message was likely deleted before timeout.
                    logger.debug(f"Message {self.message.id} not found during paginator timeout (likely deleted).")
                except discord.Forbidden:
                     logger.warning(f"Missing permissions to edit message {self.message.id} on paginator timeout.")
                except discord.HTTPException as e:
                     logger.error(f"HTTP error editing message {self.message.id} on paginator timeout: {e}")
                except Exception as e:
                     # Catch unexpected errors during edit.
                     logger.exception(f"Unexpected error editing message {self.message.id} on paginator timeout: {e}")
            else:
                # Cannot edit ephemeral messages after they expire.
                logger.debug(f"Skipping edit for ephemeral message {self.message.id} on timeout.")

        # Stop the view listener.
        self.stop()

# END OF FILE paginator.py