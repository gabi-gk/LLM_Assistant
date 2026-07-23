"""
Main application logic for the system tray assistant
- Manages the tray icon and hotkey using tkinter (actual chat GUI) and pystray (system tray icons and hotkeys)
- Loads the model and RAG in the background on startup
- Shows the chat window when the hotkey is pressed
- Handles messages from the chat window, runs RAG search and agent loop, returns responses
- Saves conversation history and session state on exit
- injects self file into chat history for Marvin only
"""

from pathlib import Path
import threading
import sys
import difflib
from PIL import Image, ImageDraw
import pystray
import keyboard
from config import DEBUG, COMPACTION_THRESHOLD, COMPACTION_KEEP_RECENT, LOGS_DIR, HOTKEY, DISCORD_ENABLED
from core.model import load_model, create_streamer
from core.history import compact_history, save_conversation, save_current_session, load_last_session, save_session_state
from core.self_management import inject_self_model
from core.rag import RAG
from agent.loop import run_agent
from tray.window import ChatWindow
from tools.notifications import restore_reminders, send_notification, set_escalation, set_tk_root
from tools.knowledge import set_rag
from tools.apps.discord_bot import set_discord_model, start_discord_bot
from core.self_management import verify_self_model, restore_self_model, snapshot_self_model

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
        self.cancel = threading.Event() # cancelation of the agent loop

    def initialise(self):
        """
        Load model and RAG - runs in background thread on startup
        """
        # escalation callback before restory so anything still not done is restored
        set_tk_root(self.window.window)

        def escalation_handler(title, message):
            """
            Inject chat message when a reminder is ignored past escalation time
            """
            def handle():
                '''
                Handle the tkinker thread to not freeze the GPU
                '''
                # pass the problem to Marvin
                trigger = (f"[REMINDER] The user has not confirmed the '{title}' "
                           f"reminder: {message}. Check in on them - see if they're stuck or need a hand.")
                self.conversation_history.append({"role": "user", "content": trigger})
                self.full_history.append({"role": "user", "content": trigger})

                # Generate his response
                reply = run_agent(self.model, self.tokenizer, self.conversation_history, self.streamer)
                self.conversation_history.append({"role": "assistant", "content": reply})
                self.full_history.append({"role": "assistant", "content": reply})

                self.window.window.after(0, lambda: (
                        self.window.show(),
                        self.window.append_message("Marvin", reply, "ai")
                    ))
            threading.Thread(target=handle, daemon=True).start()

        set_escalation(escalation_handler)

        print("[TRAY] Loading model...")
        self.model, self.tokenizer = load_model()
        self.streamer = create_streamer(self.tokenizer)

        self.rag = RAG()
        self.rag.index_all() # index any new data for RAG
        set_rag(self.rag)

        if DISCORD_ENABLED:
            set_discord_model(self.model, self.tokenizer, self.rag)
            start_discord_bot()

        restore_reminders() # check for any acive reminders and restore them

        # Self model verification and injection 
        loaded = load_last_session()
        self.full_history = loaded
        self.conversation_history = list(loaded)  # copy; compaction will trim this independently
        is_restored = bool(loaded) # whether the session was restored or not

        status = verify_self_model()
        if DEBUG:
            print(f"[SELF VERIFY] status: {status['code']}, user_changed: {status['user_changed']}")
        if status["code"] == "no_file":
            # no file found, notify the user
            print("[SELF VERIFY] Self-model file missing!")

        elif status["code"] == "no_marker":
            # file config error, notify the user
            print("[SELF VERIFY] Self-model missing OBSERVATIONS marker - misconfigured")

        elif status["observations_repeat"]:
            # model is experiencing runaway observations, restore from snapshot
            print("[SELF VERIFY] Runaway observations detected - restoring from snapshot")
            restore_self_model()

        elif status["user_changed"]:
                    snap_user, live_user = status["user_diff"]
                    diff = difflib.unified_diff(
                        snap_user.splitlines(), live_user.splitlines(), lineterm=""
                    )
                    # reformat into plain added/removed lines, skip the unified-diff headers
                    changes = []
                    for line in diff:
                        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
                            continue # skip the noisy headers
                        if line.startswith("+"):
                            changes.append(f"Added: {line[1:].strip()}")
                        elif line.startswith("-"):
                            changes.append(f"Removed: {line[1:].strip()}")
                    change_text = "\n".join(changes) if changes else "(no line-level changes)"
                    msg = (f"Self-model user section changed since last snapshot.\n 'accept self' to keep it, 'restore self' to revert.\n\n{change_text}")
                    self.window.window.after(500, lambda: self.window.append_message("System", msg, "system"))

        if status["observations_long"]:
            print("[SELF VERIFY] Observations section is getting long - consider a review")

        if status["ok"]:
            snapshot_self_model() # only snapshot a clean file
        inject_self_model(self.conversation_history, prepend=True)

        # summarise the restored history so the model gets manageable context from the start
        if is_restored:  # only fire if there is history to restore
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

    def on_cancel(self):
        """
        Called when the user presses the cancel button
        """

        self.cancel.set() # signal the agent loop to cancel

    def on_message(self, user_input):
        """
        Called by ChatWindow when user sends a message
        Runs RAG search, calls agent loop, returns response string

        user_input: string of the user's message from the chat window
        returns: string response from the assistant to be displayed in the chat window
        """
        if not self.ready:
            return "Model is still loading, please wait..."

        if user_input.lower() == "cancel":
            self.on_cancel()
            return "[INTERRUPT_SENT]"

        # handle pre-defined special commands
        if user_input.lower() == "clear":
            save_conversation(self.full_history)
            save_session_state("cleared")
            self.conversation_history = []
            self.full_history = []
            self.window.clear_display()
            inject_self_model(self.conversation_history, prepend=True)
            return "History cleared"
        
        elif user_input.lower() == "restore self":
            if restore_self_model():
                inject_self_model(self.conversation_history, prepend=True) # keep context consistent
                return "Self-model restored from the last snapshot"
            return "No snapshot available to restore from. Requires manual file fix"
        
        elif user_input.lower() == "accept self":
            snapshot_self_model() # Save the current file as the new baseline
            return "Self-model changes accepted - new baseline saved."
        
        self.conversation_history.append({"role": "user", "content": user_input})
        self.full_history.append({"role": "user", "content": user_input})
        self.cancel.clear() # clear any previous cancelation

        reply = run_agent(
            self.model, self.tokenizer,
            self.conversation_history,
            self.streamer,
            cancel=self.cancel
        )

        if reply.startswith("[INTERRUPT]"):
            self.conversation_history.pop() # remove the interrupted user message from the model history
            self.full_history.pop()
            save_session_state("closed")
            save_current_session(self.full_history)
            return reply

        self.conversation_history.append({"role": "assistant", "content": reply})
        self.full_history.append({"role": "assistant", "content": reply})

        # compact only the model context - full_history is never touched
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
        Toggle the chat window on hotkey press
        """
        if self.window:
            if self.window.visible:
                self.window.hide()
            else:
                self.window.show()

    def run(self):
        """
        Start the application
        Called from run.py after acquiring lock
        Runs the tray icon in a background thread and the chat window on the main thread
        """
        # build chat window object, do not display yet, pass the on_message callback so that when a message is sent from the UI
        # it calls the on_message function in this class to handle it
        self.window = ChatWindow(on_message_callback=self.on_message, on_cancel_callback=self.on_cancel)

        # load model in background so UI appears immediately without waiting for model to load
        # daemon thread will automatically exit the tray app when main thread exits
        threading.Thread(target=self.initialise, daemon=True).start()

        # global hotkey to show the chat window, works even when the app is not focused
        keyboard.add_hotkey(HOTKEY, self.on_show)

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