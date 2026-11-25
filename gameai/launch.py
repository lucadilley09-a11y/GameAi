"""One window, no black console: start.bat runs this with pythonw (hidden).

It installs updates, starts the app, and opens ONE app window. After an update the
app restarts quietly and the page reloads itself, so no second window appears.
If the app is already running, it just opens a window onto it.
"""
import os, subprocess, sys, time, urllib.request

PORT = 8765
URL = f"http://127.0.0.1:{PORT}/"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def running():
    try:
        urllib.request.urlopen(URL + "api/status", timeout=1.5); return True
    except Exception:
        return False


def version_id():
    try:
        return open(".version", encoding="utf-8").read()
    except OSError:
        return ""


def popup(text):
    """Show an error box (there's no console to print to)."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, "Game AI", 0x10)
    except Exception:
        print(text)


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs("logs", exist_ok=True)
    from .app import open_window
    background = "--background" in sys.argv  # started with Windows: no window, stays on for your Chromebook
    if running():
        if not background: open_window(URL)
        return
    first, updated, rolled_back = not background, False, False
    while True:
        with open("logs/app.log", "a", encoding="utf-8") as log:
            log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"); log.flush()
            before = version_id()
            subprocess.run([sys.executable, "-m", "gameai.update"], stdout=log, stderr=log, creationflags=NO_WINDOW)
            updated = updated or version_id() != before
            args = [sys.executable, "-m", "gameai.app", "--auto-quit"] + ([] if first else ["--no-window"]) + (["--stay"] if background else [])
            t0 = time.time()
            code = subprocess.run(args, stdout=log, stderr=log, creationflags=NO_WINDOW).returncode
            if code not in (0, 42) and time.time() - t0 < 60 and updated and not rolled_back:
                # the new version crashed right away: put the previous one back and start that instead
                log.write("The new version failed to start: putting the previous version back.\n"); log.flush()
                subprocess.run([sys.executable, "-m", "gameai.update", "--rollback"], stdout=log, stderr=log, creationflags=NO_WINDOW)
                rolled_back, updated = True, False
                continue
        first = False
        if code == 42:
            updated = True
            continue  # updated: restart; the open window reloads by itself
        if code != 0 and time.time() - t0 < 60:
            tail = open("logs/app.log", encoding="utf-8", errors="replace").read().splitlines()[-12:]
            popup("Game AI stopped because of an error.\nSend Claude these lines (also saved in logs\\app.log):\n\n" + "\n".join(tail))
        return


if __name__ == "__main__":
    main()
