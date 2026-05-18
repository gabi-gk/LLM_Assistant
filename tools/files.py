'''
Allows the assistant to read, write, append, and create files, as well as list directory contents
- Includes confirmation prompts before writing/appending to files
- searches for files across common user directories
- can locate a full path fron just a filename
'''
from pathlib import Path
from config import DEBUG, SEARCH_DIRS
import tkinter.messagebox as messagebox

def find_file(filename):
    """
    Search for a file by name across common user directories.

    filename: the name of the file to search for (not a path)
    returns the full path if found, or an error message if not found
    """
    for directory in SEARCH_DIRS: # search pre-defined directories for the file
        path = Path(directory)
        if not path.exists():
            continue
        matches = list(path.rglob(filename)) # add any matches to a list
        if matches:
            if DEBUG:
                print(f"[DEBUG] Found '{filename}' at {matches[0]}") # return the first match found
            return str(matches[0])

    searched = ", ".join(str(d) for d in SEARCH_DIRS if Path(d).exists())
    return f"[ERROR] '{filename}' not found. Searched: {searched}"

def read_file(path):
    """
    Read and return the contents of any file

    path: the path to the file to read, or just a filename to search for
    returns the file contents, or an error message if the file cannot be read
    """
    if not any(c in path for c in ("/", "\\")) and not Path(path).exists(): 
        found = find_file(path)
        if found.startswith("[ERROR]"): # if the file returned an error, pass that back instead of trying to read it
            return found
        if DEBUG:
            print(f"[DEBUG] Resolved '{path}' to '{found}'")
        path = found

    try:
        # Check if the path exists
        path = Path(path)
        if not path.exists():
            return f"[ERROR] File not found: {path}"
        if not path.is_file():
            return f"[ERROR] Path is not a file: {path}"
        content = path.read_text(encoding="utf-8") # read the file contents and return them
        return f"[FILE: {path}]\n{content}"
    except Exception as e:
        return f"[ERROR] Could not read file: {e}"


def write_file(path, content):
    """
    Write content to a file, replacing existing content
    Show preview before confirmation

    path: the path to the file to write, or just a filename to search for
    content: the string content to write to the file
    """
    path = Path(path)

    # Show the preview and ask for confirmation before proceeding
    print(f"\n[WRITE] Target: {path}")
    print(f"[WRITE] Content preview:\n{'-'*40}\n{content}\n{'-'*40}")
    if path.exists():
        print(f"[WRITE] This will overwrite existing file")

    if not confirm(f"Write {len(content)} chars to {path}?"): # Get the confirmation from the user
        return "[CANCELLED] Write cancelled"

    try:
        # Write to the specified file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"[SUCCESS] Written to {path}"
    except Exception as e:
        return f"[ERROR] Could not write file: {e}"


def append_file(path, content):
    """
    Append content to an existing file
    Show preview before confirmation

    path: the path to the file to append to, or just a filename to search for
    content: the string content to append to the file
    """
    path = Path(path)

    # Show the preview and ask for confirmation before proceeding
    print(f"\n[APPEND] Target: {path}")
    print(f"[APPEND] Content to append:\n{'-'*40}\n{content}\n{'-'*40}")
    if not path.exists():
        print(f"[APPEND] File does not exist, will be created")

    if not confirm(f"Append {len(content)} chars to {path}?"): # Get the confirmation from the user
        return "[CANCELLED] Append cancelled"

    try:
        # Append to the specified file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"[SUCCESS] Appended to {path}"
    except Exception as e:
        return f"[ERROR] Could not append to file: {e}"


def create_file(path, content=""):
    """
    Create a new file. Warns if file already exists
    Show preview before confirmation

    path: the path to the file to create, or just a filename to search for
    content: the string content to write to the new file (default is empty)
    returns the confirmation/cancellation messages from the write_file function
    """
    path = Path(path)
    if path.exists():
        print(f"\n[CREATE] Warning: {path} already exists")
        if not confirm(f"Overwrite {path}?"): # Get the confirmation from the user
            return "[CANCELLED] Create cancelled"
    return write_file(path, content)


def list_directory(path="."):
    """
    List files and folders in a directory

    path: the path to the directory to list, or just a directory name to search for (defaults to current directory)
    returns a formatted string listing the contents of the directory
    """
    try:
        path = Path(path)
        if not path.exists():
            return f"[ERROR] Directory not found: {path}"
        if not path.is_dir():
            return f"[ERROR] Path is not a directory: {path}"

        items = list(path.iterdir()) # get all the items in the directory
        folders = sorted([i.name + "/" for i in items if i.is_dir()]) # add a "/" after folder names to distinguish them from files, and sort alphabetically
        files = sorted([i.name for i in items if i.is_file()]) # get just the file names and sort alphabetically

        result = f"[DIR: {path.resolve()}]\n"
        result += "\n".join(folders + files) # combine the files and folders into a single list and join them into a string with line breaks
        return result
    except Exception as e:
        return f"[ERROR] Could not list directory: {e}"


def confirm(prompt, use_gui=True):
    """ 
    Ask for confirmation and return true if confirmed

    prompt: the confirmation message to show
    use_gui: true for tray, false for terminal
    returns True if the user confirms, False otherwise
    """
    if use_gui:
        return messagebox.askyesno("Marvin — Confirm", prompt)
    else:
        response = input(f"\n[CONFIRM] {prompt} (y/n): ").strip().lower()
        return response in ("y", "yes")