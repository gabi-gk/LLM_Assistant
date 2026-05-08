'''
Controls what goes in the chat window and how it looks, using tkinter for a simple terminal-style interface
- Displays user messages, AI responses, system messages and errors in different colours
'''

import tkinter as tk
from tkinter import scrolledtext
import threading

class ChatWindow:
    """
    Terminal-style dark chat window
    Hides to tray on close, never destroys the model
    """

    def __init__(self, on_message_callback):
        """
        Initialize the chat window

        on_message_callback — function called with user input string when the user sends a message
        """
        self.on_message = on_message_callback
        self.window = None
        self.input_field = None
        self.chat_display = None
        self.build()

    def build(self):
        """
        Build the tkinter window - the GUI setup is done here
        """
        self.window = tk.Tk()
        self.window.title("Marvin")
        self.window.geometry("700x500")
        self.window.configure(bg="#1e1e1e")

        # hide to tray on close instead of destroying
        self.window.protocol("WM_DELETE_WINDOW", self.hide)

        # chat display — scrollable, read only
        self.chat_display = scrolledtext.ScrolledText(
            self.window,
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=("Consolas", 11),
            wrap=tk.WORD,
            state=tk.DISABLED,
            borderwidth=0,
            highlightthickness=0,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        # colour tags for different message types
        self.chat_display.tag_config("user", foreground="#10f51b") # green
        self.chat_display.tag_config("ai", foreground="#fdfdfd") # white
        self.chat_display.tag_config("system", foreground="#222dc9") # blue
        self.chat_display.tag_config("error", foreground="#eb2727") # red
        self.chat_display.tag_config("tool", foreground="#e0e015") # yellow

        # input area at the bottom
        input_frame = tk.Frame(self.window, bg="#1e1e1e")
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # prompt label
        tk.Label(
            input_frame,
            text="You:",
            bg="#1e1e1e",
            fg="#21be29",
            font=("Consolas", 11)
        ).pack(side=tk.LEFT, padx=(0, 8))

        # text input
        self.input_field = tk.Entry(
            input_frame,
            bg="#2d2d2d",
            fg="#d4d4d4",
            font=("Consolas", 11),
            insertbackground="#d4d4d4",
            borderwidth=0,
            highlightthickness=1,
            highlightcolor="#21be29",
            highlightbackground="#3c3c3c"
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind("<Return>", self.on_enter)
        self.input_field.focus()

        # send button
        tk.Button(
            input_frame,
            text="Send",
            bg="#21be29",
            fg="white",
            font=("Consolas", 11),
            borderwidth=0,
            padx=12,
            command=self.on_enter,
            activebackground="#0dd119",
            activeforeground="white",
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=(8, 0))

        # start hidden
        self.window.withdraw()

    def on_enter(self, event=None):
        """
        Handle message submission
        """
        user_input = self.input_field.get().strip()
        if not user_input:
            return

        self.input_field.delete(0, tk.END) # delete from input field immediately for better UX
        self.append_message("You", user_input, "user") # add user message to display

        # run model in background thread so UI stays responsive
        threading.Thread(
            target=self.process_message,
            args=(user_input,),
            daemon=True
        ).start()

    def load_history(self, conversation_history):
        """
        Load a conversation history into the chat display

        conversation_history: list of dicts with 'role' and 'content' keys representing the conversation history
        """
        for message in conversation_history:
            content = message["content"]
            # skip summary messages
            if message["content"].startswith("[Earlier conversation summary:"):
                continue
            # skip tool result messages
            if message["content"].startswith("<tool_result>"):
                continue
            # skip tool call messages  
            if "<tool>" in message["content"] and "</tool>" in message["content"]:
                continue
            if message["role"] == "user":
                # compacted summary messages are synthetic so show them as system context, not user input
                if content.startswith("[Earlier conversation summary:"):
                    self.append_message("System", content, "system")
                else:
                    self.append_message("You", content, "user")
            elif message["role"] == "assistant":
                self.append_message("Marvin", content, "ai")

    def clear_display(self):
        """
        Wipe all messages from the chat display whne the user clears history
        """
        def _update():
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)

        self.window.after(0, _update)

    def process_message(self, user_input):
        """
        Run in background thread — calls model and updates display

        user_input — the message string from the user
        """
        self.append_message("System", "Thinking...", "system")
        try:
            response = self.on_message(user_input)
            # Show pure response
            self.remove_last_system()
            self.append_message("Marvin", response, "ai")
        except Exception as e:
            self.remove_last_system()
            self.append_message("Error", str(e), "error")

    def append_message(self, sender, message, tag):
        """
        Add a message to the chat display thread-safely

        sender — the name to show as the message sender
        message — the message content string
        tag — the tkinter text tag to apply for styling
        """
        def _update():
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.insert(tk.END, f"{sender}: ", tag)
            self.chat_display.insert(tk.END, f"{message}\n\n", tag)
            self.chat_display.config(state=tk.DISABLED)
            self.chat_display.see(tk.END)  # auto scroll to bottom

        # schedule on main thread since tkinter isn't thread safe
        self.window.after(0, _update)

    def remove_last_system(self):
        """
        Remove the last system message
        """
        def update():
            self.chat_display.config(state=tk.NORMAL)
            # find and delete the last "System: Thinking..." line
            content = self.chat_display.get("1.0", tk.END)
            last_system = content.rfind("System: Thinking...")
            if last_system != -1:
                # convert char index to tkinter line.col format
                line_num = content[:last_system].count("\n") + 1
                self.chat_display.delete(f"{line_num}.0", f"{line_num + 1}.0")
            self.chat_display.config(state=tk.DISABLED)

        self.window.after(0, update)

    def show(self):
        """
        Show and focus the window
        """
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self.input_field.focus()

    def hide(self):
        """
        Hide to tray — model stays loaded
        """
        self.window.withdraw()

    def run(self):
        """
        Start the tkinter main loop — must be called from main thread.
        """
        self.window.mainloop()