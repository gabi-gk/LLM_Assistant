"""
Main application logic for the system tray assistant
- Manages the tray icon and hotkey using tkinter (actual chat GUI) and pystray (system tray icons and hotkeys)
- Loads the model and RAG in the background on startup
- Shows the chat window when the hotkey is pressed
- Handles messages from the chat window, runs RAG search and agent loop, returns responses
- Saves conversation history and session state on exit
"""

from pathlib import Path
import threading
import sys
from PIL import Image, ImageDraw
import pystray
import keyboard
from config import DEBUG, COMPACTION_THRESHOLD, COMPACTION_KEEP_RECENT, LOGS_DIR
from core.model import load_model, create_streamer
from core.history import compact_history, save_conversation, save_current_session, load_last_session, save_session_state
from core.rag import RAG
from agent.loop import run_agent
from tray.window import ChatWindow
from tools.notifications import restore_reminders, send_notification
from tools.knowledge import set_rag
from tools.apps.discord_bot import set_discord_model, start_discord_bot

class TrayApp:
    """
    Main application class
    Manages the tray icon, hotkey, chat window and model lifecycle
    """

    def __init__(self):
        """
        Initialize variables
        """
        self.model = None
        self.tokenizer = None
        self.streamer = None
        self.rag = None
        self.conversation_history = []
        self.full_history = []
        self.window = None
        self.tray = None
        self.ready = False

    def initialise(self):
        """
        Load model and RAG — runs in background thread on startup
        """
        print("[TRAY] Loading model...")
        self.model, self.tokenizer = load_model()
        self.streamer = create_streamer(self.tokenizer)

        self.rag = RAG()
        self.rag.index_all() # index any new data for RAG
        set_rag(self.rag)

        set_discord_model(self.model, self.tokenizer, self.rag)
        start_discord_bot()

        restore_reminders() # check for any acive reminders and restore them

        loaded = load_last_session()
        self.full_history = loaded
        self.conversation_history = list(loaded)  # copy; compaction will trim this independently

        if self.conversation_history:
            # summarise the restored history so the model gets manageable context from the start
            self.conversation_history = compact_history(
                self.model, self.tokenizer,
                self.conversation_history,
                threshold=COMPACTION_THRESHOLD,
                keep_recent=COMPACTION_KEEP_RECENT
            )
            self.window.load_history(self.full_history)
            self.window.append_message("System", "Previous session restored.", "system")

        self.ready = True
        print("[TRAY] Ready.")

        # notify user model is loaded
        send_notification("I'm ready to Chat!", "Model 'Marvin' loaded and ready to use.")

    def on_message(self, user_input):
        """
        Called by ChatWindow when user sends a message
        Runs RAG search, calls agent loop, returns response string

        user_input: string of the user's message from the chat window
        returns: string response from the assistant to be displayed in the chat window
        """
        if not self.ready:
            return "Model is still loading, please wait..."

        # handle pre-defined special commands
        if user_input.lower() == "clear":
            save_conversation(self.full_history)
            save_session_state("cleared")
            self.conversation_history = []
            self.full_history = []
            self.window.clear_display()
            return "History cleared"

        self.conversation_history.append({"role": "user", "content": user_input})
        self.full_history.append({"role": "user", "content": user_input})

        reply = run_agent(
            self.model, self.tokenizer,
            self.conversation_history,
            self.streamer
        )

        self.conversation_history.append({"role": "assistant", "content": reply})
        self.full_history.append({"role": "assistant", "content": reply})

        # compact only the model context — full_history is never touched
        self.conversation_history = compact_history(
            self.model, self.tokenizer,
            self.conversation_history,
            threshold=COMPACTION_THRESHOLD,
            keep_recent=COMPACTION_KEEP_RECENT
        )

        save_session_state("closed")
        save_current_session(self.full_history)

        return reply

    def create_tray_icon(self):
        """
        Create a simple coloured circle as the tray icon
        returns an image object
        """
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([8, 8, 56, 56], fill="#10f51b")
        draw.ellipse([20, 20, 44, 44], fill="white")
        return img

    def on_quit(self):
        """
        Save conversation as a timestamp and fully quit
        """
        save_conversation(self.full_history)  # creates timestamped file
        save_session_state("closed")
        
        # remove the current file so it can be used by the next session
        current = Path(f"{LOGS_DIR}/current_session.json")
        if current.exists():
            current.unlink()

        self.tray.stop()
        sys.exit(0)

    def on_show(self):
        """
        Show the chat window on hotkey press, if it's not already visible
        """
        if self.window:
            self.window.show()

    def run(self):
        """
        Start the application
        Called from run.py after acquiring lock
        Runs the tray icon in a background thread and the chat window on the main thread
        """
        # build chat window object, do not display yet, pass the on_message callback so that when a message is sent from the UI
        # it calls the on_message function in this class to handle it
        self.window = ChatWindow(on_message_callback=self.on_message)

        # load model in background so UI appears immediately without waiting for model to load
        # daemon thread will automatically exit the tray app when main thread exits
        threading.Thread(target=self.initialise, daemon=True).start()

        # register hotkey - add to config later?
        # global hotkey to show the chat window, works even when the app is not focused
        keyboard.add_hotkey("alt+space", self.on_show)

        # build right click menu
        menu = pystray.Menu(
            pystray.MenuItem("Open", lambda icon, item: self.on_show(), default=True), # Add Open option to the right-click menu to show the chat window as a default action (bolded)
            # lambda used so the pystray required parameters are passed but the function which doesn't need them, does not need to see them
            pystray.MenuItem("Quit", lambda icon, item: self.on_quit())
        )
        self.tray = pystray.Icon(
            "assistant", # unique ID for the tray icon, not displayed
            self.create_tray_icon(), # the icon image
            "Marvin", # tooltip text
            menu # right click menu
        )

        # run tray in background thread, solve pystray and tkinker conflict by running them on separate threads
        # pystray will run in tghe background and tkinter will run on the main thread
        threading.Thread(
            target=self.tray.run,
            daemon=True
        ).start()

        # run tkinter on main thread
        self.window.run()