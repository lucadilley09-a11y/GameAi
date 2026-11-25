"""Running one part of the AI from another (for example gameai.all running gameai.train) so that its output
really reaches the app.

Why this exists: on Windows, when the app runs a job without a window, a job's own sub-jobs used to print into
nowhere: the training's progress never reached the app (the bar sat at 5% the whole time even though training
was going fine), nothing reached logs/job.log, and when a sub-job failed its error message was lost too.
Here the sub-job's output is read and passed on line by line instead.
"""
import os, subprocess, sys


def run(args, prefix=None):
    """python -m <args>, its output passed on to ours; -> exit code"""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen([sys.executable, "-u", "-m", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                         text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
    except KeyboardInterrupt:
        p.terminate(); raise
    return p.wait()
