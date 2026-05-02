from pathlib import Path
from config import DEBUG, SEARCH_DIRS

def find_file(filename):
    """
    Search for a file by name across common user directories.
    Uses Path.home() so it works on both Windows and Linux automatically.
    """
    for directory in SEARCH_DIRS:
        path = Path(directory)
        if not path.exists():
            continue
        matches = list(path.rglob(filename))
        if matches:
            if DEBUG:
                print(f"[DEBUG] Found '{filename}' at {matches[0]}")
            return str(matches[0])

    searched = ", ".join(str(d) for d in SEARCH_DIRS if Path(d).exists())
    return f"[ERROR] '{filename}' not found. Searched: {searched}"

def read_file(path):
    """
    Read and return the contents of any file
    """
    if not any(c in path for c in ("/", "\\")) and not Path(path).exists():
        found = find_file(path)
        if found.startswith("[ERROR]"):
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
        content = path.read_text(encoding="utf-8")
        return f"[FILE: {path}]\n{content}"
    except Exception as e:
        return f"[ERROR] Could not read file: {e}"


def write_file(path, content):
    """
    Write content to a file, replacing existing content
    Show preview before confirmation
    """
    path = Path(path)

    # Show the preview and ask for confirmation before proceeding
    print(f"\n[WRITE] Target: {path}")
    print(f"[WRITE] Content preview:\n{'-'*40}\n{content}\n{'-'*40}")
    if path.exists():
        print(f"[WRITE] This will overwrite existing file")

    if not confirm(f"Write {len(content)} chars to {path}?"):
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
    """
    path = Path(path)

    # Show the preview and ask for confirmation before proceeding
    print(f"\n[APPEND] Target: {path}")
    print(f"[APPEND] Content to append:\n{'-'*40}\n{content}\n{'-'*40}")
    if not path.exists():
        print(f"[APPEND] File does not exist, will be created")

    if not confirm(f"Append {len(content)} chars to {path}?"):
        return "[CANCELLED] Append cancelled"

    try:
        # Write to the specified file
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
    """
    path = Path(path)
    if path.exists():
        print(f"\n[CREATE] Warning: {path} already exists")
        if not confirm(f"Overwrite {path}?"):
            return "[CANCELLED] Create cancelled"
    return write_file(path, content)


def list_directory(path="."):
    """
    List files and folders in a directory
    """
    try:
        path = Path(path)
        if not path.exists():
            return f"[ERROR] Directory not found: {path}"
        if not path.is_dir():
            return f"[ERROR] Path is not a directory: {path}"

        items = list(path.iterdir())
        folders = sorted([i.name + "/" for i in items if i.is_dir()])
        files = sorted([i.name for i in items if i.is_file()])

        result = f"[DIR: {path.resolve()}]\n"
        result += "\n".join(folders + files)
        return result
    except Exception as e:
        return f"[ERROR] Could not list directory: {e}"


def confirm(prompt):
    """ 
    Ask for confirmation and return true if confirmed
    """
    response = input(f"\n[CONFIRM] {prompt} (y/n): ").strip().lower()
    return response in ("y", "yes")