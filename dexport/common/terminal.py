"""Terminal keys for the deploy loop and the calibration jog.

:class:`KeyPoller` reads keys without blocking while the terminal is in cbreak mode (no line
buffering or echo; Ctrl-C still raises KeyboardInterrupt). Without a terminal (piped or nohup
stdin) it reads nothing and says so once. :func:`wait_for_key` blocks for one key at a prompt.
Both read the terminal's file descriptor directly: through ``sys.stdin``'s buffer, the rest of an
escape sequence would stay unseen by ``select`` until the next key press.
"""

from __future__ import annotations

import codecs
import os
import select
import sys

ESC = "\x1b"
_ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}


class KeyPoller:
    """Context manager with two views of the pending input. ``read()`` returns the raw characters:
    the deploy loop ends an episode on any ``ESC`` among them, i.e. on Esc and on every key that
    sends an escape sequence (an arrow or function key). ``keys()`` decodes them for the jog:
    letters lowercased, the arrow sequences as 'up' / 'down' / 'left' / 'right', Esc itself as
    'esc' and any other three-character escape sequence as 'other' (a sequence still arriving
    waits for its rest)."""

    def __init__(self):
        self._fd = None
        self._old = None
        self._buf = ""
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def __enter__(self) -> "KeyPoller":
        import termios
        import tty

        if not sys.stdin.isatty():
            print("[keys] stdin is not a terminal: Esc and Space are disabled.", flush=True)
            return self
        try:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception as e:  # e.g. no controlling terminal
            print(
                f"[keys] cannot use the terminal ({e!r}): Esc and Space are disabled.", flush=True
            )
            self._fd = None
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._fd = None

    def read(self) -> str:
        if self._fd is None:
            return ""
        data = b""
        while select.select([self._fd], [], [], 0)[0]:
            chunk = os.read(self._fd, 1024)
            if not chunk:
                break
            data += chunk
        return self._utf8.decode(data)

    def keys(self) -> list[str]:
        self._buf += self.read()
        return self._decode()

    def _decode(self) -> list[str]:
        out, b = [], self._buf
        while b:
            if b[0] != ESC:
                out.append(b[0].lower())
                b = b[1:]
            elif len(b) == 1 or b[1] not in "[O":  # Esc itself (a sequence arrives in one read)
                out.append("esc")
                b = b[1:]
            elif len(b) == 2:
                break  # the rest of a sequence is still arriving
            else:
                out.append(_ARROWS.get(b[2], "other"))
                b = b[3:]
        self._buf = b
        return out


def wait_for_key(prompt: str) -> str:
    """Print ``prompt`` and block for one key, returned lowercased. Without a terminal, the first
    character of the next input line; "" at end of input or for an empty line."""
    print(prompt, flush=True)
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip()[:1].lower()
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        key = os.read(fd, 64)  # the whole key, so no rest of an escape sequence is left behind
        return key.decode("utf-8", errors="replace")[:1].lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
