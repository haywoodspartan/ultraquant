#!/usr/bin/env bash
# One-command UltraQuant launcher for Linux / macOS.
# The Windows equivalents are UltraQuant.bat and ultraquant.ps1.
#
#   ./ultraquant.sh            the desktop app if Tk is present, else the TUI
#   ./ultraquant.sh tui        the terminal UI, always
#   ./ultraquant.sh gui        the desktop app, always
#   ./ultraquant.sh chat       plain terminal chat
#   ./ultraquant.sh test       the full suite
#   ./ultraquant.sh forge ...  build a model
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
    done
fi
if [ -z "$PY" ]; then
    echo "No Python found. Install Python 3.10+ and re-run." >&2
    exit 1
fi

if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "UltraQuant needs Python 3.10 or newer; found $("$PY" --version)" >&2
    exit 1
fi

MODE="${1:-auto}"
[ $# -gt 0 ] && shift || true

has_tk() { "$PY" -c 'import tkinter' >/dev/null 2>&1; }

case "$MODE" in
    auto)
        if has_tk; then exec "$PY" -m ultraquant.gui "$@"; fi
        echo "Tkinter not installed - starting the terminal UI instead."
        echo "(for the desktop app: apt install python3-tk, or dnf install python3-tkinter)"
        exec "$PY" -m ultraquant.tui "$@"
        ;;
    gui)
        if ! has_tk; then
            echo "Tkinter is not installed. Try: apt install python3-tk" >&2
            exit 1
        fi
        exec "$PY" -m ultraquant.gui "$@" ;;
    tui)   exec "$PY" -m ultraquant.tui "$@" ;;
    chat)  exec "$PY" -m ultraquant.interpreter.chat "$@" ;;
    forge) exec "$PY" -m ultraquant.forge.build "$@" ;;
    bench) exec "$PY" -m ultraquant.bench "$@" ;;
    test)  exec "$PY" -m unittest discover -s tests "$@" ;;
    *)     exec "$PY" -m ultraquant.gui "$MODE" "$@" ;;
esac
