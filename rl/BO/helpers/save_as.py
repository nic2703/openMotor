import tkinter as tk
from tkinter import filedialog, messagebox
from motorlib.motor import Motor
from uilib.fileIO import saveFile, fileTypes

def save_motor_as_ric(motor: Motor) -> str | None:
    """
    Open a Save As dialog on Windows and save a Motor object
    into a .ric file using the app's native saveFile() format.

    Returns the saved file path, or None if the user cancels.

    Notes:
      - Uses the same serialization format as FileManager.save().
      - Ensures the file has a .ric extension even if omitted.
    """
    if not isinstance(motor, Motor):
        raise TypeError("Argument must be a Motor instance")

    # Hidden Tkinter root window (just for the file dialog)
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()

    try:
        path = filedialog.asksaveasfilename(
            title="Save Motor As .ric",
            defaultextension=".ric",
            filetypes=[("Motor Files", "*.ric"), ("All Files", "*.*")],
            confirmoverwrite=True,
        )

        if not path:  # User cancelled
            return None

        if not path.lower().endswith(".ric"):
            path += ".ric"

        # Get the motor's dictionary and save using the app’s saveFile()
        data = motor.getDict()
        saveFile(path, data, fileTypes.MOTOR)
        print("Saved Motor configuration to ", path)

        return path

    except Exception as e:
        try:
            messagebox.showerror("Save Failed", f"Could not save file:\n{e}")
        except Exception:
            pass
        raise
    finally:
        root.destroy()
