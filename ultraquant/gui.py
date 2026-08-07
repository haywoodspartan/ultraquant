"""One-click desktop front end for UltraQuant.

A Tkinter application (stdlib only, like the rest of the project) that exposes the
whole system without a terminal:

* **Chat** — the predefined thought pipeline, colon-commands, and the thought
  trace of the last turn.
* **Forge** — build a model from scratch and watch it happen.
* **Library** — the shard catalog, what is resident, and the RAM budget.
* **Stash** — triage web claims: promote what is corroborated, reject the rest.
* **Panel** — the LLMLS: ask local LM Studio models, grouped by *independent
  lineage* rather than by headcount, with the voice count shown before the
  question is asked. Answers land in the Stash like anything else found rather
  than derived.
* **Benchmark** — measure the execution tiers on this machine.

Everything slow runs on a worker thread and reports back through a queue, because
Tk may only be touched from the thread that owns it; the UI stays responsive
while a forge or a benchmark runs.

Launch with ``UltraQuant.bat``, or ``python -m ultraquant.gui``.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - headless install
    tk = None  # type: ignore[assignment]

from ultraquant.native.dispatch import QUANTUM_TIERS
from ultraquant.storage.ram import available_ram, total_ram

__all__ = ["UltraQuantGUI", "main"]

#: Labels for the quantum tiers, in the order dispatch prefers them.
QUANTUM_TIER_LABELS: list[str] = [label for _key, label in QUANTUM_TIERS]

_POLL_MS = 80
_FONT_MONO = ("Consolas", 10)

#: Training devices offered in the Compute tab, as (menu label, trainer tier).
TRAIN_DEVICES: list[tuple[str, str]] = [
    ("Auto (pick the fastest)", "auto"),
    ("GPU and CPU together", "both"),
    ("GPU only (CUDA)", "cuda"),
    ("CPU only (C++ threads)", "cpu"),
    ("Pure Python (reference)", "python"),
]

#: Storage backends offered in the Storage tab, as (menu label, URI scheme).
#: Ordered by how likely they are to be wanted, not alphabetically.
BACKEND_CHOICES: list[tuple[str, str]] = [
    ("Local filesystem", "local"),
    ("Block volume (tuned)", "blockdev"),
    ("NVMe-oF namespace", "nvmeof"),
    ("Lightbits (NVMe/TCP)", "lightbits"),
    ("Pure Storage FlashArray", "pure"),
    ("IBM / HPE 3PAR", "3par"),
    ("CephFS mount", "cephfs"),
    ("Ceph RADOS pool", "rados"),
    ("RAM (volatile scratch)", "ram"),
    ("Custom URI...", "custom"),
]

#: Per-backend form behaviour: what to call the main field, whether a second
#: field is needed, whether the block-tuning row applies, and a one-line hint.
BACKEND_FORM: dict[str, dict] = {
    "local": {
        "label": "Directory:", "browse": True, "tuning": False,
        "hint": "Ordinary buffered files. The OS page cache helps here, so "
                "direct I/O is not used.",
    },
    "blockdev": {
        "label": "Directory:", "browse": True, "tuning": True,
        "hint": "A mounted volume with host-side tuning. Use this for a SAN LUN "
                "that is not one of the named arrays below.",
    },
    "nvmeof": {
        "label": "Mount path:", "browse": True, "tuning": True,
        "hint": "NVMe-oF namespace. Deep queues and unbuffered reads: the fabric "
                "carries many reads at once, and the host page cache mostly adds a copy.",
    },
    "lightbits": {
        "label": "Mount path:", "browse": True, "tuning": True,
        "hint": "Lightbits NVMe/TCP volume. Same host tuning as NVMe-oF; use the "
                "control plane below to provision and snapshot it.",
    },
    "pure": {
        "label": "Mount path:", "browse": True, "tuning": True,
        "hint": "Pure Storage FlashArray volume. The array has its own large cache, "
                "so host readahead is modest and double-caching is avoided.",
    },
    "3par": {
        "label": "Mount path:", "browse": True, "tuning": True,
        "hint": "IBM/HPE 3PAR volume. Larger readahead by default, since its "
                "minimum efficient transfer is bigger.",
    },
    "cephfs": {
        "label": "Mount path:", "browse": True, "tuning": False,
        "hint": "A mounted CephFS tree. POSIX semantics over a network, so deeper "
                "queues but no alignment rules.",
    },
    "rados": {
        "label": "Pool:", "browse": False, "tuning": False, "extra": "Namespace:",
        "hint": "Native RADOS objects with real ranged reads - no filesystem at all. "
                "Needs the 'rados' Python bindings (ceph-common).",
    },
    "ram": {
        "label": "Name:", "browse": False, "tuning": False,
        "hint": "In-process memory. Volatile - for scratch and experiments, never "
                "for a library you want to keep.",
    },
    "custom": {
        "label": "URI:", "browse": False, "tuning": False,
        "hint": "Type a full URI, including any ?query options.",
    },
}


class _QueueStream:
    """File-like object that funnels writes to the UI thread via a queue."""

    def __init__(self, sink: queue.Queue, tag: str = "out") -> None:
        """Bind the stream to a queue.

        Args:
            sink: Queue drained by the UI thread.
            tag: Routing tag telling the UI which pane the text belongs to.
        """
        self.sink = sink
        self.tag = tag

    def write(self, text: str) -> int:
        """Queue ``text`` for display."""
        if text:
            self.sink.put((self.tag, text))
        return len(text)

    def flush(self) -> None:
        """No-op; the queue is always current."""


class UltraQuantGUI:
    """The main window."""

    def __init__(self, root, home: Path) -> None:
        """Build the window.

        Args:
            root: The Tk root.
            home: Session directory for memory, vault, stash and archive.
        """
        self.root = root
        self.home = Path(home)
        self.events: queue.Queue = queue.Queue()
        self.session = None
        self.cli = None
        self.busy = False
        self._alive = True
        self._pump_id: str | None = None
        self.last_error: str | None = None
        #: model id -> ModelCard for the Panel tab's current catalogue.
        self._panel_cards: dict = {}

        root.title("UltraQuant")
        root.geometry("1080x720")
        root.minsize(880, 560)

        self._build_menu()
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self._build_chat_tab()
        self._build_learn_tab()
        self._build_compute_tab()
        self._build_forge_tab()
        self._build_library_tab()
        self._build_storage_tab()
        self._build_stash_tab()
        self._build_panel_tab()
        self._build_bench_tab()
        self._build_status_bar()

        self.root.bind("<Destroy>", self._on_destroy)
        self.root.after(_POLL_MS, self._pump)
        self._run_async("Starting session", self._start_session)

    # ------------------------------------------------------------------ chrome

    def _build_menu(self) -> None:
        """Menu bar."""
        menu = tk.Menu(self.root)
        session = tk.Menu(menu, tearoff=0)
        session.add_command(label="Change session folder...", command=self._choose_home)
        session.add_command(label="Consolidate (pack + snapshot)", command=self._consolidate)
        session.add_separator()
        session.add_command(label="Quit", command=self.root.destroy)
        menu.add_cascade(label="Session", menu=session)

        helpmenu = tk.Menu(menu, tearoff=0)
        helpmenu.add_command(label="About", command=self._about)
        menu.add_cascade(label="Help", menu=helpmenu)
        self.root.config(menu=menu)

    def _build_status_bar(self) -> None:
        """Status strip along the bottom."""
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", side="bottom", padx=8, pady=6)
        self.status = tk.StringVar(value="starting...")
        ttk.Label(bar, textvariable=self.status).pack(side="left")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        self.progress.pack(side="right")
        self.tiers = tk.StringVar(value="detecting devices...")
        ttk.Label(bar, textvariable=self.tiers).pack(side="right", padx=12)

    # -------------------------------------------------------------------- tabs

    def _build_chat_tab(self) -> None:
        """Chat/Interpreter tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Chat")

        panes = ttk.PanedWindow(frame, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(panes)
        panes.add(left, weight=3)
        self.transcript = self._text(left, height=20)
        self.transcript.pack(fill="both", expand=True)

        entry_row = ttk.Frame(left)
        entry_row.pack(fill="x", pady=(6, 0))
        # Pack the button before the entry: an expanding widget packed first
        # claims the whole row and squeezes anything after it off the edge.
        ttk.Button(entry_row, text="Send  (Ctrl+Enter)", command=self._send, width=18
                   ).pack(side="right", fill="y", padx=(6, 0))
        self.chat_input = tk.Text(entry_row, height=3, font=_FONT_MONO, wrap="word")
        self.chat_input.pack(side="left", fill="both", expand=True)
        self.chat_input.bind("<Control-Return>", lambda _e: self._send())

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=(6, 0))
        for label, command in (
            (":help", ":help"), ("facts", ":facts"), ("shards", ":shards"),
            ("resident", ":resident"), ("stash", ":stash"), ("snapshot", ":snapshot"),
        ):
            ttk.Button(buttons, text=label, width=10,
                       command=lambda c=command: self._submit(c)).pack(side="left", padx=2)

        right = ttk.Frame(panes)
        panes.add(right, weight=2)
        ttk.Label(right, text="Thought trace (last turn)").pack(anchor="w")
        self.trace = self._text(right, height=20)
        self.trace.pack(fill="both", expand=True)

    def _build_learn_tab(self) -> None:
        """Learning mode: the model asks, you answer."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Learn")

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="Find gaps", command=self._survey).pack(side="left")
        ttk.Button(top, text="Skip", command=self._skip_question).pack(side="left", padx=6)
        ttk.Label(
            top,
            text="   The model inspects what it holds and asks about its weakest points.",
        ).pack(side="left")

        ask = ttk.LabelFrame(frame, text="Question")
        ask.pack(fill="x", padx=6, pady=(0, 6))
        self.question_var = tk.StringVar(value="Press 'Find gaps' to begin.")
        ttk.Label(ask, textvariable=self.question_var, wraplength=980,
                  justify="left").pack(anchor="w", padx=8, pady=(8, 4))

        self.option_row = ttk.Frame(ask)
        self.option_row.pack(fill="x", padx=8, pady=(0, 4))

        answer_row = ttk.Frame(ask)
        answer_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(answer_row, text="Answer", command=self._answer_question, width=12
                   ).pack(side="right", padx=(6, 0))
        self.answer_input = tk.Text(answer_row, height=5, font=_FONT_MONO, wrap="word")
        self.answer_input.pack(side="left", fill="both", expand=True)
        self.answer_input.bind("<Control-Return>", lambda _e: self._answer_question())

        self.learn_log = self._text(frame, height=16)
        self.learn_log.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def _build_compute_tab(self) -> None:
        """Which processors do the work, and how much memory they may use."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Compute")

        # -- classical devices
        devices = ttk.LabelFrame(frame, text="Training device")
        devices.pack(fill="x", padx=6, pady=6)
        row = ttk.Frame(devices)
        row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(row, text="Use:", width=14).pack(side="left")
        self.compute_device = tk.StringVar(value=TRAIN_DEVICES[0][0])
        ttk.Combobox(row, textvariable=self.compute_device, width=34, state="readonly",
                     values=[label for label, _tier in TRAIN_DEVICES]).pack(side="left")
        ttk.Label(row, text="CPU threads:").pack(side="left", padx=(16, 2))
        self.compute_threads = tk.IntVar(value=0)
        ttk.Spinbox(row, from_=0, to=max(1, os.cpu_count() or 1), width=6,
                    textvariable=self.compute_threads).pack(side="left")
        ttk.Label(row, text=f"(0 = all {os.cpu_count() or 1})").pack(side="left", padx=4)

        self.compute_devices_label = tk.StringVar(value="detecting...")
        ttk.Label(devices, textvariable=self.compute_devices_label, wraplength=1000,
                  justify="left").pack(anchor="w", padx=8, pady=(0, 8))

        # -- quantum tier
        quantum = ttk.LabelFrame(frame, text="Quantum backend")
        quantum.pack(fill="x", padx=6, pady=(0, 6))
        qrow = ttk.Frame(quantum)
        qrow.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(qrow, text="Run circuits on:", width=14).pack(side="left")
        self.quantum_tier = tk.StringVar(value=QUANTUM_TIER_LABELS[0])
        ttk.Combobox(qrow, textvariable=self.quantum_tier, width=34, state="readonly",
                     values=QUANTUM_TIER_LABELS).pack(side="left")
        ttk.Label(qrow, text="Shots:").pack(side="left", padx=(16, 2))
        self.quantum_shots = tk.StringVar(value="0")
        ttk.Entry(qrow, textvariable=self.quantum_shots, width=8).pack(side="left")
        ttk.Label(qrow, text="(0 = exact expectation, no sampling)").pack(side="left", padx=4)
        ttk.Button(qrow, text="Test backend", command=self._test_quantum
                   ).pack(side="left", padx=12)

        crow = ttk.Frame(quantum)
        crow.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(crow, text="BlueQubit token:", width=14).pack(side="left")
        self.bluequbit_token = tk.StringVar(value=os.environ.get("BLUEQUBIT_API_TOKEN", ""))
        ttk.Entry(crow, textvariable=self.bluequbit_token, width=34, show="*").pack(side="left")
        ttk.Button(crow, text="Use token", command=self._apply_bluequbit_token
                   ).pack(side="left", padx=8)
        ttk.Label(crow, text="(cloud GPU/MPS/QPU at app.bluequbit.io; kept in this "
                             "process only)").pack(side="left", padx=4)

        self.quantum_label = tk.StringVar(value="detecting...")
        ttk.Label(quantum, textvariable=self.quantum_label, wraplength=1000,
                  justify="left").pack(anchor="w", padx=8, pady=(0, 8))

        # -- memory
        memory = ttk.LabelFrame(frame, text="Memory")
        memory.pack(fill="x", padx=6, pady=(0, 6))
        total_mb = max(1, (total_ram() or 0) // (1024 * 1024))
        avail_mb = max(1, (available_ram() or 0) // (1024 * 1024))
        self._ram_total_mb = total_mb
        self._ram_avail_mb = avail_mb

        mrow = ttk.Frame(memory)
        mrow.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(mrow, text="RAM tier:", width=14).pack(side="left")
        self.ram_mb = tk.DoubleVar(value=min(1024, max(64, avail_mb // 4)))
        ttk.Scale(mrow, from_=64, to=max(128, avail_mb), orient="horizontal",
                  variable=self.ram_mb, length=560,
                  command=lambda _v: self._update_ram_label()).pack(side="left")
        self.ram_label = tk.StringVar()
        ttk.Label(mrow, textvariable=self.ram_label, width=34).pack(side="left", padx=8)

        ttk.Label(memory, wraplength=1000, justify="left", text=(
            f"This machine has {total_mb:,} MB of RAM, {avail_mb:,} MB free. The slider "
            f"is capped at what is currently available, so the working set can never "
            f"claim memory the system does not have. The library itself stays on storage."
        )).pack(anchor="w", padx=8, pady=(0, 4))

        brow = ttk.Frame(memory)
        brow.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(brow, text="Apply to session", command=self._apply_compute
                   ).pack(side="left")
        ttk.Button(brow, text="Refresh detection", command=self._detect_compute
                   ).pack(side="left", padx=6)

        self.compute_log = self._text(frame, height=12)
        self.compute_log.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._update_ram_label()

    def _build_storage_tab(self) -> None:
        """Storage backends, the RAM tier, and array control planes."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Storage")

        backend = ttk.LabelFrame(frame, text="Shard library backend")
        backend.pack(fill="x", padx=6, pady=6)

        # -- pick the medium, then fill in only what that medium needs
        row = ttk.Frame(backend)
        row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(row, text="Backend:", width=10).pack(side="left")
        self.storage_backend = tk.StringVar(value=BACKEND_CHOICES[0][0])
        picker = ttk.Combobox(
            row, textvariable=self.storage_backend, width=42, state="readonly",
            values=[label for label, _scheme in BACKEND_CHOICES],
        )
        picker.pack(side="left")
        picker.bind("<<ComboboxSelected>>", lambda _e: self._on_backend_change())
        ttk.Label(row, text="RAM tier:").pack(side="left", padx=(16, 2))
        self.storage_cache = tk.StringVar(value="auto")
        ttk.Combobox(row, textvariable=self.storage_cache, width=10, values=[
            "auto", "off", "64MB", "256MB", "1GB", "4GB", "16GB",
        ]).pack(side="left")
        ttk.Button(row, text="Inspect", command=self._inspect_storage).pack(side="left", padx=(16, 4))
        ttk.Button(row, text="Scaling analysis", command=self._scale_analysis).pack(side="left")

        row2 = ttk.Frame(backend)
        row2.pack(fill="x", padx=8, pady=4)
        self.storage_location_label = tk.StringVar(value="Directory:")
        ttk.Label(row2, textvariable=self.storage_location_label, width=10).pack(side="left")
        self.storage_location = tk.StringVar(value="./uq_home/vault")
        self.storage_location.trace_add("write", lambda *_a: self._update_uri_preview())
        ttk.Entry(row2, textvariable=self.storage_location, width=42).pack(side="left")
        self.storage_browse = ttk.Button(row2, text="Browse...", command=self._browse_storage)
        self.storage_browse.pack(side="left", padx=4)

        self.storage_extra_label = tk.StringVar(value="Namespace:")
        self.storage_extra_frame = ttk.Frame(row2)
        ttk.Label(self.storage_extra_frame, textvariable=self.storage_extra_label
                  ).pack(side="left", padx=(10, 2))
        self.storage_extra = tk.StringVar()
        self.storage_extra.trace_add("write", lambda *_a: self._update_uri_preview())
        ttk.Entry(self.storage_extra_frame, textvariable=self.storage_extra, width=16
                  ).pack(side="left")

        # -- tuning that only applies to block-device backends
        self.storage_tuning = ttk.Frame(backend)
        self.storage_tuning.pack(fill="x", padx=8, pady=4)
        ttk.Label(self.storage_tuning, text="Tuning:", width=10).pack(side="left")
        self.storage_direct = tk.BooleanVar(value=True)
        # Trace rather than the widget's command, so the preview stays truthful
        # however the value changes.
        self.storage_direct.trace_add("write", lambda *_a: self._update_uri_preview())
        self.storage_direct_box = ttk.Checkbutton(
            self.storage_tuning, text="Unbuffered (direct) I/O",
            variable=self.storage_direct,
        )
        self.storage_direct_box.pack(side="left")
        ttk.Label(self.storage_tuning, text="Queue depth:").pack(side="left", padx=(14, 2))
        self.storage_qdepth = tk.StringVar(value="")
        self.storage_qdepth.trace_add("write", lambda *_a: self._update_uri_preview())
        self.storage_qdepth_entry = ttk.Entry(
            self.storage_tuning, textvariable=self.storage_qdepth, width=6
        )
        self.storage_qdepth_entry.pack(side="left")
        ttk.Label(self.storage_tuning, text="Readahead (KB):").pack(side="left", padx=(14, 2))
        self.storage_readahead = tk.StringVar(value="")
        self.storage_readahead.trace_add("write", lambda *_a: self._update_uri_preview())
        self.storage_readahead_entry = ttk.Entry(
            self.storage_tuning, textvariable=self.storage_readahead, width=6
        )
        self.storage_readahead_entry.pack(side="left")

        # -- the URI actually built, so nothing about the choice is hidden
        row4 = ttk.Frame(backend)
        row4.pack(fill="x", padx=8, pady=(4, 4))
        ttk.Label(row4, text="URI:", width=10).pack(side="left")
        self.storage_uri = tk.StringVar(value="local://./uq_home/vault")
        ttk.Entry(row4, textvariable=self.storage_uri, width=78, state="readonly"
                  ).pack(side="left")

        self.storage_hint = tk.StringVar()
        ttk.Label(backend, textvariable=self.storage_hint, wraplength=1000,
                  justify="left").pack(anchor="w", padx=8, pady=(0, 8))
        self._on_backend_change()

        locations = ttk.LabelFrame(frame, text="Locations")
        locations.pack(fill="x", padx=6, pady=(0, 6))

        lib_row = ttk.Frame(locations)
        lib_row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(lib_row, text="Library:", width=10).pack(side="left")
        self.library_root = tk.StringVar(value=str(self.home / "vault"))
        ttk.Entry(lib_row, textvariable=self.library_root, width=58).pack(side="left")
        ttk.Button(lib_row, text="Browse...",
                   command=lambda: self._browse_into(self.library_root,
                                                     "Choose the shard library directory")
                   ).pack(side="left", padx=4)

        forge_row = ttk.Frame(locations)
        forge_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(forge_row, text="Forge out:", width=10).pack(side="left")
        self.forge_root = tk.StringVar(value=str(self.home / "forged"))
        ttk.Entry(forge_row, textvariable=self.forge_root, width=58).pack(side="left")
        ttk.Button(forge_row, text="Browse...",
                   command=lambda: self._browse_into(self.forge_root,
                                                     "Choose the forge output directory")
                   ).pack(side="left", padx=4)

        loc_actions = ttk.Frame(locations)
        loc_actions.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(loc_actions, text="Apply locations", command=self._apply_locations
                   ).pack(side="left")
        ttk.Button(loc_actions, text="Open session here", command=self._reopen_session
                   ).pack(side="left", padx=6)
        ttk.Label(loc_actions, wraplength=760, justify="left", text=(
            "   The library is where shards live and are paged from; the forge "
            "directory is where newly built models are written."
        )).pack(side="left")

        array = ttk.LabelFrame(frame, text="Array control plane (provisioning and snapshots)")
        array.pack(fill="x", padx=6, pady=(0, 6))
        arow = ttk.Frame(array)
        arow.pack(fill="x", padx=8, pady=6)
        ttk.Label(arow, text="Vendor:").pack(side="left")
        self.array_vendor = tk.StringVar(value="pure")
        ttk.Combobox(arow, textvariable=self.array_vendor, width=10, state="readonly",
                     values=["pure", "3par", "lightbits"]).pack(side="left", padx=4)
        ttk.Label(arow, text="Endpoint:").pack(side="left", padx=(10, 2))
        self.array_endpoint = tk.StringVar()
        ttk.Entry(arow, textvariable=self.array_endpoint, width=30).pack(side="left")
        ttk.Label(arow, text="User:").pack(side="left", padx=(10, 2))
        self.array_user = tk.StringVar()
        ttk.Entry(arow, textvariable=self.array_user, width=12).pack(side="left")
        ttk.Label(arow, text="Token/Password:").pack(side="left", padx=(10, 2))
        self.array_secret = tk.StringVar()
        ttk.Entry(arow, textvariable=self.array_secret, width=18, show="*").pack(side="left")

        arow2 = ttk.Frame(array)
        arow2.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(arow2, text="Connect", command=self._array_connect).pack(side="left")
        ttk.Button(arow2, text="List volumes", command=self._array_volumes
                   ).pack(side="left", padx=6)
        ttk.Button(arow2, text="Capacity", command=self._array_capacity).pack(side="left")
        ttk.Label(arow2, text="  Volume:").pack(side="left", padx=(12, 2))
        self.array_volume = tk.StringVar()
        ttk.Entry(arow2, textvariable=self.array_volume, width=18).pack(side="left")
        ttk.Button(arow2, text="Snapshot library", command=self._array_snapshot
                   ).pack(side="left", padx=6)
        self.array_insecure = tk.BooleanVar(value=False)
        ttk.Checkbutton(arow2, text="Skip TLS verify", variable=self.array_insecure
                        ).pack(side="left", padx=10)

        self.storage_log = self._text(frame, height=18)
        self.storage_log.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def _build_forge_tab(self) -> None:
        """Model forge tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Forge")

        controls = ttk.LabelFrame(frame, text="Build a model from scratch")
        controls.pack(fill="x", padx=6, pady=6)

        self.forge_vars = {
            "source": tk.StringVar(value="synthetic"),
            "categories": tk.StringVar(value="64"),
            "labels": tk.StringVar(value="4"),
            "per_class": tk.StringVar(value="40"),
            "epochs": tk.StringVar(value="40"),
            "hidden": tk.StringVar(value="32"),
            "tier": tk.StringVar(value="auto"),
        }

        row = ttk.Frame(controls)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Patterns:").pack(side="left")
        ttk.Radiobutton(row, text="Built-in glyph families", value="builtin",
                        variable=self.forge_vars["source"]).pack(side="left", padx=6)
        ttk.Radiobutton(row, text="Invent new ones", value="synthetic",
                        variable=self.forge_vars["source"]).pack(side="left", padx=6)

        row2 = ttk.Frame(controls)
        row2.pack(fill="x", padx=8, pady=6)
        for label, key, width in (
            ("Categories", "categories", 6), ("Labels/cat", "labels", 4),
            ("Variants", "per_class", 5), ("Epochs", "epochs", 5),
            ("Hidden", "hidden", 5),
        ):
            ttk.Label(row2, text=f"{label}:").pack(side="left", padx=(10, 2))
            ttk.Entry(row2, textvariable=self.forge_vars[key], width=width).pack(side="left")
        ttk.Label(row2, text="Tier:").pack(side="left", padx=(10, 2))
        ttk.Combobox(row2, textvariable=self.forge_vars["tier"], width=8, state="readonly",
                     values=["auto", "both", "cuda", "cpu", "python"]).pack(side="left")

        row3 = ttk.Frame(controls)
        row3.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(row3, text="Forge model", command=self._forge).pack(side="left")
        ttk.Button(row3, text="Compare tiers", command=lambda: self._forge(compare=True)
                   ).pack(side="left", padx=6)
        ttk.Label(row3, text="  Output goes to <session>/forged/").pack(side="left")

        self.forge_log = self._text(frame, height=22)
        self.forge_log.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def _build_library_tab(self) -> None:
        """Shard catalog and residency tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Library")

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="Refresh", command=self._refresh_library).pack(side="left")
        ttk.Label(top, text="   RAM budget (KB):").pack(side="left")
        self.budget_var = tk.StringVar(value="1024")
        ttk.Entry(top, textvariable=self.budget_var, width=8).pack(side="left", padx=4)
        ttk.Button(top, text="Apply", command=self._apply_budget).pack(side="left")
        ttk.Button(top, text="Attach .uql library...", command=self._attach
                   ).pack(side="left", padx=12)
        ttk.Button(top, text="Consolidate", command=self._consolidate).pack(side="left")

        columns = ("shard", "category", "where", "bytes", "uses")
        self.catalog = ttk.Treeview(frame, columns=columns, show="headings", height=14)
        for column, width in zip(columns, (280, 130, 90, 100, 70)):
            self.catalog.heading(column, text=column)
            self.catalog.column(column, width=width,
                                anchor="e" if column in ("bytes", "uses") else "w")
        self.catalog.pack(fill="both", expand=True, padx=6)

        self.residency = self._text(frame, height=7)
        self.residency.pack(fill="x", padx=6, pady=6)

    def _build_stash_tab(self) -> None:
        """Contemporary stash triage tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Stash")

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="URL:").pack(side="left")
        self.url_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.url_var, width=52).pack(side="left", padx=4)
        self.online_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Allow network", variable=self.online_var,
                        command=self._toggle_online).pack(side="left", padx=6)
        ttk.Button(top, text="Fetch", command=self._fetch).pack(side="left")
        ttk.Button(top, text="Refresh", command=self._refresh_stash).pack(side="left", padx=6)

        columns = ("id", "class", "status", "sources", "claim")
        self.stash_view = ttk.Treeview(frame, columns=columns, show="headings", height=14)
        for column, width in zip(columns, (50, 110, 110, 70, 620)):
            self.stash_view.heading(column, text=column)
            self.stash_view.column(column, width=width,
                                   anchor="e" if column in ("id", "sources") else "w")
        self.stash_view.pack(fill="both", expand=True, padx=6)

        actions = ttk.Frame(frame)
        actions.pack(fill="x", padx=6, pady=6)
        ttk.Button(actions, text="Analyze", command=self._analyze).pack(side="left")
        ttk.Button(actions, text="Promote to fact", command=self._promote).pack(side="left", padx=6)
        ttk.Button(actions, text="Promote (force)",
                   command=lambda: self._promote(force=True)).pack(side="left")
        ttk.Button(actions, text="Reject", command=self._reject).pack(side="left", padx=6)
        ttk.Label(
            actions,
            text="   Nothing fetched becomes a fact until it is corroborated or promoted here.",
        ).pack(side="left")

    def _build_panel_tab(self) -> None:
        """The LLMLS teacher panel: local models, grouped by independent voice.

        The tree is the argument. Models are shown *under the voice they
        belong to*, so a panel of four Llama derivatives is visibly one source
        rather than four, and the running selection label says so as you pick.
        That accounting is the reason this tab exists; a flat list of model
        names would hide exactly the thing worth seeing.
        """
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Panel")

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="Refresh catalogue",
                   command=self._panel_refresh).pack(side="left")
        ttk.Button(top, text="Load selected",
                   command=lambda: self._panel_load(True)).pack(side="left", padx=6)
        ttk.Button(top, text="Unload selected",
                   command=lambda: self._panel_load(False)).pack(side="left")
        self.panel_status = tk.StringVar(value="LM Studio: not checked")
        ttk.Label(top, textvariable=self.panel_status).pack(side="left", padx=12)

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=True, padx=6)
        self.panel_tree = ttk.Treeview(tree_frame, columns=("state", "arch",
                                                            "publisher"),
                                       selectmode="extended", height=11)
        self.panel_tree.heading("#0", text="voice / model")
        for column, width in (("state", 90), ("arch", 130), ("publisher", 130)):
            self.panel_tree.heading(column, text=column)
            self.panel_tree.column(column, width=width, anchor="w")
        self.panel_tree.column("#0", width=440)
        scroll = ttk.Scrollbar(tree_frame, command=self.panel_tree.yview)
        self.panel_tree.configure(yscrollcommand=scroll.set)
        self.panel_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.panel_tree.bind("<<TreeviewSelect>>", self._panel_selection_changed)

        selection = ttk.Frame(frame)
        selection.pack(fill="x", padx=6, pady=(4, 0))
        self.panel_selection = tk.StringVar(
            value="Select models above. Agreement counts by voice, not headcount.")
        ttk.Label(selection, textvariable=self.panel_selection,
                  wraplength=980, justify="left").pack(anchor="w")

        ask = ttk.LabelFrame(frame, text="Question")
        ask.pack(fill="x", padx=6, pady=6)
        row = ttk.Frame(ask)
        row.pack(fill="x", padx=8, pady=8)
        self.panel_question = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.panel_question)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self._panel_ask())
        ttk.Button(row, text="Ask panel", command=self._panel_ask, width=12
                   ).pack(side="left", padx=(6, 0))
        self.panel_to_stash = tk.BooleanVar(value=True)
        ttk.Checkbutton(ask, text="Quarantine answers in the stash "
                        "(nothing becomes a fact here)",
                        variable=self.panel_to_stash).pack(anchor="w", padx=8,
                                                           pady=(0, 8))

        self.panel_log = self._text(frame, height=14)
        self.panel_log.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def _build_bench_tab(self) -> None:
        """Benchmark tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Benchmark")

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="Run tier benchmark", command=self._benchmark).pack(side="left")
        ttk.Button(top, text="Shard paging demo", command=self._paging_demo
                   ).pack(side="left", padx=6)
        ttk.Label(top, text="   Compares pure Python, C++, CUDA and CPU+GPU."
                  ).pack(side="left")

        self.bench_log = self._text(frame, height=26)
        self.bench_log.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    @staticmethod
    def _text(parent, height: int = 18) -> tk.Text:
        """A read-only monospace text pane with a scrollbar."""
        wrapper = ttk.Frame(parent)
        widget = tk.Text(wrapper, height=height, font=_FONT_MONO, wrap="word",
                         state="disabled", background="#101317", foreground="#d8dee9",
                         insertbackground="#d8dee9")
        scroll = ttk.Scrollbar(wrapper, command=widget.yview)
        widget.configure(yscrollcommand=scroll.set)
        widget.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        wrapper.pack(fill="both", expand=True)
        widget.master_frame = wrapper  # type: ignore[attr-defined]
        return widget

    # ------------------------------------------------------------- plumbing

    def _append(self, widget: tk.Text, text: str) -> None:
        """Append text to a read-only pane and scroll to the end."""
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _set(self, widget: tk.Text, text: str) -> None:
        """Replace a pane's contents."""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _on_destroy(self, event) -> None:
        """Stop polling once the window itself goes away.

        The pending timer must be cancelled, not merely flagged: Tk fires an
        already-queued ``after`` callback during teardown, from inside its own
        event loop where no Python guard of ours runs, and complains about the
        deleted command on stderr.
        """
        if event.widget is not self.root:
            return
        self._alive = False
        if self._pump_id is not None:
            try:
                self.root.after_cancel(self._pump_id)
            except tk.TclError:  # pragma: no cover - interpreter already gone
                pass
            self._pump_id = None

    def _notify(self, message: str) -> None:
        """Report something to the user without seizing the window.

        Modal dialogs are avoided everywhere except the About box: this app
        drives long jobs from worker threads, and a modal raised off the back of
        one blocks the event loop that the job needs in order to report progress.
        """
        self.status.set(message)
        self._append(self.transcript, f"[ui] {message}\n")

    def _run_async(self, label: str, work: Callable[[], None]) -> None:
        """Run ``work`` on a worker thread, keeping the UI alive.

        Args:
            label: Status text shown while it runs.
            work: Callable executed off the UI thread.
        """
        if self.busy:
            self._notify("Busy - wait for the current task to finish.")
            return
        self.busy = True
        self.status.set(label + "...")
        self.progress.start(12)

        def runner() -> None:
            try:
                work()
            except Exception:  # noqa: BLE001 - surfaced in the UI, never a crash
                self.events.put(("error", traceback.format_exc()))
            finally:
                self.events.put(("done", label))

        threading.Thread(target=runner, daemon=True, name="uq-worker").start()

    def _pump(self) -> None:
        """Drain the worker queue on the UI thread.

        Every widget touch is guarded: the timer can still fire once after the
        window has been destroyed, and an unguarded callback would spew Tcl
        errors on exit.
        """
        self._pump_id = None
        try:
            self._pump_once()
        except tk.TclError:
            self._alive = False
        if self._alive:
            try:
                self._pump_id = self.root.after(_POLL_MS, self._pump)
            except tk.TclError:  # pragma: no cover - window gone mid-schedule
                self._alive = False

    def _pump_once(self) -> None:
        """Drain every queued event exactly once."""
        try:
            while True:
                tag, payload = self.events.get_nowait()
                if tag == "out":
                    self._append(self.transcript, payload)
                elif tag == "forge":
                    self._append(self.forge_log, payload)
                elif tag == "bench":
                    self._append(self.bench_log, payload)
                elif tag == "learn":
                    self._append(self.learn_log, payload)
                elif tag == "panel":
                    self._append(self.panel_log, payload)
                elif tag == "panel_status":
                    self.panel_status.set(payload)
                elif tag == "panel_tree":
                    self._fill_panel_tree(payload)
                elif tag == "panel_reload":
                    # Deferred: this event is queued *before* the worker's
                    # "done", so busy is still set and _run_async would
                    # refuse with a spurious "Busy" popup. One tick later
                    # the done event has been drained.
                    self.root.after(_POLL_MS + 1, self._panel_refresh)
                elif tag == "storage":
                    self._append(self.storage_log, payload)
                elif tag == "devices":
                    self.compute_devices_label.set(payload)
                elif tag == "quantum":
                    self.quantum_label.set(payload)
                elif tag == "locations":
                    self.library_root.set(str(Path(payload) / "vault"))
                    self.forge_root.set(str(Path(payload) / "forged"))
                elif tag == "compute":
                    self._append(self.compute_log, payload)
                elif tag == "question":
                    self._show_question()
                elif tag == "trace":
                    self._set(self.trace, payload)
                elif tag == "status":
                    self.status.set(payload)
                elif tag == "tiers":
                    self.tiers.set(payload)
                elif tag == "refresh":
                    self._refresh_library()
                    self._refresh_stash()
                elif tag == "error":
                    # Reported in place rather than through a modal: a background
                    # failure should not seize the window, and the full traceback
                    # is more useful than a one-line popup.
                    self._append(self.transcript, f"\n[error]\n{payload}\n")
                    self.last_error = payload
                    self.notebook.select(0)
                elif tag == "done":
                    self.busy = False
                    self.progress.stop()
                    self.status.set(
                        "error - see the Chat pane" if self.last_error else "ready"
                    )
                    self.last_error = None
        except queue.Empty:
            pass

    # -------------------------------------------------------------- actions

    def _start_session(self) -> None:
        """Build the interpreter session and report available tiers."""
        from ultraquant.forge.trainer import forge_tier_report
        from ultraquant.interpreter.chat import ChatCLI
        from ultraquant.interpreter.thoughts import build_session

        self.session = build_session(self.home, budget_bytes=1024 * 1024, seed=0)
        self.cli = ChatCLI(self.session, out=_QueueStream(self.events, "out"))

        try:
            from ultraquant.native.dispatch import tier_report
            native = tier_report()
        except Exception:  # noqa: BLE001 - native tiers are optional
            native = {}
        try:
            forge = forge_tier_report()
        except Exception:  # noqa: BLE001 - a probe failure must not blank the strip
            forge = {}

        parts = [f"CPU: {os.cpu_count() or '?'} cores"]
        gpu = native.get("cuda")
        parts.append(f"GPU: {gpu}" if isinstance(gpu, str)
                     else ("GPU: CUDA" if gpu else "GPU: none"))
        if native.get("native_cpu"):
            parts.append("C++")
        if forge.get("cuda"):
            parts.append("forge-GPU")
        parts.append("QPU" if native.get("qpu") else "QPU: none")
        self.events.put(("tiers", " | ".join(parts)))
        self.events.put(("locations", str(self.home)))
        self.events.put(("out", f"UltraQuant session at {self.home}\n"
                                f"Type below, or use ':help'. Ctrl+Enter sends.\n\n"))
        self.events.put(("refresh", None))
        # Fill the Compute tab now, so it never sits on "detecting..." until
        # somebody thinks to press Refresh.
        self.events.put(("devices", " | ".join(parts)))
        quantum = [
            "QPU reachable" if native.get("qpu") else "QPU: not configured",
            "CUDA simulator" if native.get("cuda") else "CUDA simulator: absent",
            "C++ simulator" if native.get("native_cpu") else "C++ simulator: absent",
            "Python simulator (always)",
        ]
        self.events.put(("quantum", " | ".join(quantum)))

    def _send(self) -> str:
        """Send whatever is in the input box."""
        text = self.chat_input.get("1.0", "end").rstrip("\n")
        self.chat_input.delete("1.0", "end")
        if text.strip():
            self._submit(text)
        return "break"

    def _submit(self, text: str) -> None:
        """Run one input through the CLI on a worker thread."""
        if self.cli is None:
            return
        self._append(self.transcript, f"> {text}\n")

        def work() -> None:
            lines = text.splitlines()
            if text.lstrip().startswith(":"):
                # A colon-command may consume following lines (":teach" wants five
                # glyph rows), so hand it the rest of the block as its input feed.
                stream = iter(lines[1:])
                self.cli.handle(lines[0], stream)
                for extra in stream:
                    if extra.strip():
                        self.cli.handle(extra, stream)
            else:
                # Plain input is one thing, however many lines it spans -- a pasted
                # glyph is five rows that only mean anything together.
                self.cli.handle(text)
            trace = getattr(self.session, "last_trace", []) or []
            rendered = "\n".join(f"{s['thought']:<9} {s['summary']}" for s in trace)
            self.events.put(("trace", rendered or "(no trace for this input)"))
            self.events.put(("out", "\n"))
            self.events.put(("refresh", None))

        self._run_async("Thinking", work)

    def _forge(self, compare: bool = False) -> None:
        """Forge a model with the current settings."""
        try:
            categories = int(self.forge_vars["categories"].get())
            labels = int(self.forge_vars["labels"].get())
            per_class = int(self.forge_vars["per_class"].get())
            epochs = int(self.forge_vars["epochs"].get())
            hidden = int(self.forge_vars["hidden"].get())
        except ValueError:
            self._notify("Forge settings must be whole numbers.")
            return
        # The Compute tab owns device choice; the Forge tab's own selector
        # stays as an override for a one-off run.
        tier = self.forge_vars["tier"].get()
        if tier == "auto":
            # "auto" stays "auto": resolving it here to a concrete device used
            # to bypass the learned dispatcher entirely, so the scheduler
            # never saw the workload. An explicit device choice on the
            # Compute tab is still an override.
            selected = self._selected_train_tier()
            tier = selected if selected != "auto" else "auto"
        threads = int(self.compute_threads.get() or 0)
        synthetic = self.forge_vars["source"].get() == "synthetic"
        out_root = Path(self.forge_root.get().strip() or (self.home / "forged"))

        def work() -> None:
            from ultraquant.forge.build import main as forge_main

            argv = ["--root", str(out_root), "--epochs", str(epochs),
                    "--per-class", str(per_class), "--hidden", str(hidden),
                    "--tier", tier, "--threads", str(threads)]
            if synthetic:
                argv += ["--synthetic", str(categories), "--labels", str(labels)]
            if compare:
                argv.append("--compare")

            stream = _QueueStream(self.events, "forge")
            saved = sys.stdout
            sys.stdout = stream  # the forge CLI prints its own progress
            try:
                self.events.put(("forge", f"\n$ forge {' '.join(argv)}\n\n"))
                forge_main(argv)
            finally:
                sys.stdout = saved
            self.events.put(("forge", "\n"))
            self.events.put(("refresh", None))

        self._run_async("Forging model", work)

    def _benchmark(self) -> None:
        """Run the tier benchmark."""
        def work() -> None:
            from ultraquant.bench import main as bench_main

            stream = _QueueStream(self.events, "bench")
            saved = sys.stdout
            sys.stdout = stream
            try:
                self.events.put(("bench", "\n$ python -m ultraquant.bench\n\n"))
                bench_main(["--samples", "512", "--big-qubits", "18"])
            finally:
                sys.stdout = saved

        self._run_async("Benchmarking", work)

    def _paging_demo(self) -> None:
        """Run the on-demand shard paging demonstration."""
        def work() -> None:
            from ultraquant.shards.scale_demo import main as demo_main

            stream = _QueueStream(self.events, "bench")
            saved = sys.stdout
            sys.stdout = stream
            try:
                self.events.put(("bench", "\n$ python -m ultraquant.shards.scale_demo\n\n"))
                demo_main(["--shards", "512", "--budget-kb", "64", "--accesses", "400"])
            finally:
                sys.stdout = saved

        self._run_async("Paging demo", work)

    # ------------------------------------------------------ learning mode

    # ----------------------------------------------------------- compute

    def _dispatch_summary(self) -> list[str]:
        """What the learned dispatcher knows, for the Compute tab."""
        import json as _json

        lines: list[str] = []
        for where in (self.home / "forged" / "dispatch.json",
                      self.home / "dispatch.json"):
            if not where.exists():
                continue
            try:
                payload = _json.loads(where.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt file is not a crash
                continue
            records = len(payload.get("records", []))
            lines.append(f"learned dispatch ({where.parent.name}): "
                         f"{records} timings recorded")
            for kind, verdict in sorted(payload.get("verdicts", {}).items()):
                lines.append(
                    f"  {kind}: brain={verdict.get('chosen')} "
                    f"acc={verdict.get('accuracy')} "
                    f"decide_us={verdict.get('decide_us')}"
                )
        return lines or ["learned dispatch: no experience yet - forge with "
                         "tier 'auto' to start probing"]

    def _selected_train_tier(self) -> str:
        """Trainer tier for the chosen training device."""
        label = self.compute_device.get()
        for choice, tier in TRAIN_DEVICES:
            if choice == label:
                return tier
        return "auto"

    def _selected_quantum_tier(self) -> str:
        """Quantum tier key for the chosen backend."""
        from ultraquant.native.dispatch import QUANTUM_TIERS

        label = self.quantum_tier.get()
        for key, text in QUANTUM_TIERS:
            if text == label:
                return key
        return "auto"

    def _update_ram_label(self) -> None:
        """Show the slider value in human units and as a share of the machine."""
        megabytes = int(self.ram_mb.get())
        share = 100.0 * megabytes / max(self._ram_total_mb, 1)
        if megabytes >= 1024:
            pretty = f"{megabytes / 1024:.1f} GB"
        else:
            pretty = f"{megabytes} MB"
        self.ram_label.set(f"{pretty}  ({share:.1f}% of system RAM)")

    def _detect_compute(self) -> None:
        """Report which processors and quantum tiers this machine has."""
        def work() -> None:
            from ultraquant.forge.trainer import forge_tier_report
            from ultraquant.native.dispatch import tier_report

            forge = forge_tier_report()
            native = tier_report()
            gpu = native.get("cuda")
            parts = [f"CPU: {os.cpu_count() or '?'} logical cores"]
            parts.append(f"GPU: {gpu}" if isinstance(gpu, str) else
                         ("GPU: CUDA device present" if gpu else "GPU: none"))
            parts.append("training tiers: " + ", ".join(
                k for k, v in forge.items() if v) or "python only")
            self.events.put(("devices", " | ".join(parts)))

            from ultraquant.native.dispatch import cloud_report

            cloud = cloud_report()
            quantum = []
            quantum.append("QPU reachable" if native.get("qpu") else "QPU: not configured")
            quantum.append("BlueQubit cloud ready" if cloud.get("bluequbit")
                           else "BlueQubit: no token/SDK")
            quantum.append("CUDA simulator" if native.get("cuda") else "CUDA simulator: absent")
            quantum.append("C++ simulator" if native.get("native_cpu") else "C++ simulator: absent")
            quantum.append("Python simulator (always)")
            self.events.put(("quantum", " | ".join(quantum)))
            for line in self._dispatch_summary():
                self.events.put(("compute", line + "\n"))
            self.events.put((
                "compute",
                f"\nDetected devices\n  {' | '.join(parts)}\n  {' | '.join(quantum)}\n",
            ))

        self._run_async("Detecting devices", work)

    def _test_quantum(self) -> None:
        """Run a Bell state on the chosen backend and show the result."""
        tier = self._selected_quantum_tier()
        try:
            shots = int(self.quantum_shots.get() or 0)
        except ValueError:
            self._notify("Shots must be a whole number (0 = exact).")
            return

        def work() -> None:
            from ultraquant.native.dispatch import backend_by_name
            from ultraquant.quantum.circuit import Circuit

            backend = backend_by_name(tier, seed=0)
            circuit = Circuit(2).h(0).cnot(0, 1)
            result = backend.run(circuit, shots=shots or None)
            expectations = ", ".join(f"{v:+.6f}" for v in result.expectations_z)
            self.events.put((
                "compute",
                f"\n$ backend={tier!r} shots={shots or 'exact'}\n"
                f"  using       : {backend.name}\n"
                f"  Bell <Z>    : [{expectations}]   (both should be 0)\n"
                + (f"  counts      : {result.counts}\n" if result.counts else ""),
            ))
            if tier != "auto" and tier not in backend.name.replace("-", ""):
                self.events.put((
                    "compute",
                    f"  note        : '{tier}' was not available, so the run fell "
                    f"back to {backend.name}. Every tier computes the same values.\n",
                ))

        self._run_async("Testing quantum backend", work)

    def _apply_bluequbit_token(self) -> None:
        """Make a BlueQubit token available to this process.

        Set in the environment rather than written anywhere: the token is a
        credential, and nothing here should persist it to disk.
        """
        token = self.bluequbit_token.get().strip()
        if not token:
            os.environ.pop("BLUEQUBIT_API_TOKEN", None)
            self._notify("BlueQubit token cleared.")
            self._detect_compute()
            return
        os.environ["BLUEQUBIT_API_TOKEN"] = token
        self._append(self.compute_log,
                     "\nBlueQubit token set for this session (not written to disk).\n")
        self._detect_compute()

    def _apply_compute(self) -> None:
        """Apply the compute choices to the running session."""
        megabytes = int(self.ram_mb.get())
        self.storage_cache.set(f"{megabytes}MB")
        if self.session is not None:
            self.session.cache.set_budget(megabytes * 1024 * 1024)
            storage = getattr(self.session, "storage", None)
            if storage is not None and hasattr(storage, "set_budget"):
                storage.set_budget(megabytes * 1024 * 1024)
        self._append(self.compute_log, (
            f"\nApplied:\n"
            f"  training   : {self._selected_train_tier()} "
            f"(threads: {self.compute_threads.get() or 'all'})\n"
            f"  quantum    : {self._selected_quantum_tier()} "
            f"(shots: {self.quantum_shots.get() or 'exact'})\n"
            f"  RAM tier   : {megabytes} MB\n"
        ))
        self._refresh_library()

    def _survey(self) -> None:
        """Look for knowledge gaps and show the best question."""
        if self.session is None:
            return

        def work() -> None:
            from ultraquant.interpreter.learning import LearningSession

            self.learner = LearningSession(self.session)
            found = self.learner.survey()
            self.events.put(("learn", f"\nFound {len(found)} gap(s):\n"))
            for question in found:
                self.events.put((
                    "learn",
                    f"  [{question.id}] {question.kind:<20} score {question.score:5.2f}\n",
                ))
            self.events.put(("question", None))

        self._run_async("Looking for gaps", work)

    def _show_question(self) -> None:
        """Display the next question and its options."""
        learner = getattr(self, "learner", None)
        for child in self.option_row.winfo_children():
            child.destroy()
        if learner is None:
            return
        question = learner.next_question()
        if question is None:
            self.question_var.set("Nothing left to ask. Press 'Find gaps' again later.")
            return
        rows = question.context.get("rows") or []
        if rows:
            # The prompt embeds the glyph, and a proportional wrapped label turns
            # it into noise. Show the words in the label and the shape in a
            # fixed-width one, or the user cannot see what is being asked about.
            lines = [ln for ln in question.prompt.splitlines()
                     if ln.strip() not in rows]
            self.question_var.set(f"[{question.kind}]  " + " ".join(
                line.strip() for line in lines if line.strip()))
            tk.Label(self.option_row, text="\n".join(rows), font=("Consolas", 11),
                     justify="left", anchor="w").pack(anchor="w", pady=(0, 4))
        else:
            self.question_var.set(f"[{question.kind}]  {question.prompt}")

        if question.expects == "glyph":
            ttk.Label(self.option_row,
                      text="Answer with five rows of five, using # and .").pack(anchor="w")
        elif question.kind == "unknown-pattern":
            ttk.Label(self.option_row,
                      text="Answer as '<category> <label>', e.g. 'diagonals backslash'."
                      ).pack(anchor="w")
        for option in question.options:
            ttk.Button(
                self.option_row, text=option, width=max(10, len(option) + 2),
                command=lambda o=option: self._answer_question(o),
            ).pack(side="left", padx=(0, 6))

    def _answer_question(self, preset: str | None = None) -> None:
        """Apply an answer to the current question."""
        learner = getattr(self, "learner", None)
        if learner is None:
            self._notify("Press 'Find gaps' first.")
            return
        question = learner.next_question()
        if question is None:
            self._notify("No question is waiting.")
            return
        reply = preset if preset is not None else self.answer_input.get("1.0", "end").strip()
        if not reply:
            self._notify("Type an answer, or press Skip.")
            return
        self.answer_input.delete("1.0", "end")

        def work() -> None:
            answer = learner.answer(question, reply)
            mark = "learned" if answer.accepted else "not applied"
            self.events.put((
                "learn",
                f"\n  Q: {question.prompt}\n  A: {reply[:120]}\n  -> {mark}: {answer.detail}\n",
            ))
            self.events.put(("question", None))
            self.events.put(("refresh", None))

        self._run_async("Learning", work)

    def _skip_question(self) -> None:
        """Set the current question aside."""
        learner = getattr(self, "learner", None)
        if learner is None:
            return
        question = learner.next_question()
        if question is not None:
            learner.skip(question.id)
            self._append(self.learn_log, f"  skipped [{question.id}] {question.kind}\n")
        self._show_question()

    # ----------------------------------------------------------- storage

    def _selected_scheme(self) -> str:
        """The URI scheme for the chosen backend."""
        label = self.storage_backend.get()
        for choice_label, scheme in BACKEND_CHOICES:
            if choice_label == label:
                return scheme
        return "local"

    def _on_backend_change(self) -> None:
        """Reshape the form for the selected backend."""
        scheme = self._selected_scheme()
        form = BACKEND_FORM.get(scheme, BACKEND_FORM["local"])

        self.storage_location_label.set(form["label"])
        self.storage_hint.set(form["hint"])

        self.storage_browse.state(["!disabled"] if form["browse"] else ["disabled"])

        if form.get("extra"):
            self.storage_extra_label.set(form["extra"])
            self.storage_extra_frame.pack(side="left")
        else:
            self.storage_extra_frame.pack_forget()

        # Tuning applies only where unbuffered aligned I/O is meaningful.
        state = ["!disabled"] if form["tuning"] else ["disabled"]
        for widget in (self.storage_direct_box, self.storage_qdepth_entry,
                       self.storage_readahead_entry):
            widget.state(state)

        defaults = {
            "local": "./uq_home/vault", "blockdev": "D:/uq", "nvmeof": "D:/uq",
            "lightbits": "D:/uq", "pure": "D:/uq", "3par": "D:/uq",
            "cephfs": "/mnt/cephfs/uq", "rados": "models", "ram": "scratch",
            "custom": "local://./uq_home/vault",
        }
        current = self.storage_location.get().strip()
        if not current or current in defaults.values():
            self.storage_location.set(defaults.get(scheme, ""))
        self._update_uri_preview()

    def _update_uri_preview(self) -> None:
        """Rebuild the URI from the form and show it."""
        try:
            self.storage_uri.set(self._compose_uri())
        except (AttributeError, tk.TclError):  # pragma: no cover - during build
            pass

    def _compose_uri(self) -> str:
        """Build a storage URI from the current form state."""
        scheme = self._selected_scheme()
        location = self.storage_location.get().strip()
        if scheme == "custom":
            return location
        if scheme == "rados":
            namespace = self.storage_extra.get().strip()
            return f"rados://{location}" + (f"/{namespace}" if namespace else "")

        uri = f"{scheme}://{location}"
        form = BACKEND_FORM.get(scheme, {})
        if not form.get("tuning"):
            return uri

        options = []
        if not self.storage_direct.get():
            options.append("direct=0")
        depth = self.storage_qdepth.get().strip()
        if depth.isdigit() and int(depth) > 0:
            options.append(f"queue_depth={int(depth)}")
        readahead = self.storage_readahead.get().strip()
        if readahead.isdigit() and int(readahead) > 0:
            options.append(f"readahead={int(readahead) * 1024}")
        return uri + ("?" + "&".join(options) if options else "")

    def _browse_into(self, variable, title: str) -> None:
        """Pick a directory into ``variable``."""
        chosen = filedialog.askdirectory(title=title)
        if chosen:
            variable.set(chosen)

    def _apply_locations(self) -> None:
        """Point the backend URI at the chosen library directory."""
        library = self.library_root.get().strip()
        if not library:
            self._notify("Set a library directory first.")
            return
        self.storage_backend.set("Local filesystem")
        self._on_backend_change()
        self.storage_location.set(library)
        self._append(self.storage_log, (
            f"\nLocations set:\n"
            f"  library  : {library}\n"
            f"  forge out: {self.forge_root.get().strip()}\n"
            f"  URI now  : {self._compose_uri()}\n"
            f"  Use 'Open session here' to move the running session to it.\n"
        ))

    def _reopen_session(self) -> None:
        """Reopen the session against the chosen library location.

        The library directory is a vault; its parent is the session folder that
        also holds memory, the stash and the archive, so the session moves as a
        whole rather than leaving its knowledge behind.
        """
        library = Path(self.library_root.get().strip() or (self.home / "vault"))
        home = library.parent if library.name == "vault" else library
        self.home = home
        self._append(self.storage_log, f"\nReopening session at {home}\n")
        self._run_async("Starting session", self._start_session)

    def _browse_storage(self) -> None:
        """Pick the directory backing the library."""
        chosen = filedialog.askdirectory(title="Choose the shard library directory")
        if chosen:
            self.storage_location.set(chosen)

    def _inspect_storage(self) -> None:
        """Open the configured backend and report what it can do."""
        uri = self._compose_uri().strip()
        cache = self.storage_cache.get().strip()
        if not uri or uri.endswith("://"):
            self._notify("Fill in the backend location first.")
            return

        def work() -> None:
            import json

            from ultraquant.storage import StorageError, open_storage

            self.events.put(("storage", f"\n$ open_storage({uri!r}, cache={cache!r})\n"))
            try:
                storage = open_storage(uri, cache=None if cache == "off" else cache)
            except OSError as exc:
                # A drive or mount that is not present is the single most likely
                # thing to go wrong here, and it deserves a sentence rather than
                # a stack trace.
                self.events.put((
                    "storage",
                    f"  Cannot use that location: {exc.strerror or exc}\n"
                    f"  '{uri.split('://', 1)[-1]}' is not reachable on this machine. "
                    f"Check the volume is mounted, or pick another backend.\n",
                ))
                return
            except StorageError as exc:
                self.events.put(("storage", f"  {exc}\n"))
                return
            try:
                self.events.put((
                    "storage", json.dumps(storage.describe(), indent=2) + "\n"
                ))
            finally:
                storage.close()

        self._run_async("Opening storage", work)

    def _scale_analysis(self) -> None:
        """Run the trillion-parameter scaling analysis."""
        def work() -> None:
            from ultraquant.storage.scale import main as scale_main

            stream = _QueueStream(self.events, "storage")
            saved = sys.stdout
            sys.stdout = stream
            try:
                self.events.put(("storage", "\n$ python -m ultraquant.storage.scale\n"))
                scale_main(["--build", "100000", "--lookups", "1000"])
            finally:
                sys.stdout = saved

        self._run_async("Scaling analysis", work)

    def _array(self):
        """Build a controller from the form.

        Raises:
            ArrayError: If the vendor or fields are wrong.
        """
        from ultraquant.storage.vendors import controller_for

        vendor = self.array_vendor.get()
        common = {
            "endpoint": self.array_endpoint.get().strip(),
            "insecure": bool(self.array_insecure.get()),
        }
        secret = self.array_secret.get()
        if vendor == "pure":
            return controller_for("pure", api_token=secret, **common)
        if vendor == "3par":
            return controller_for(
                "3par", username=self.array_user.get().strip(), password=secret, **common
            )
        return controller_for("lightbits", jwt=secret, **common)

    def _array_call(self, label: str, fn) -> None:
        """Run one array operation and print its result."""
        if not self.array_endpoint.get().strip():
            self._notify("Enter the array's management endpoint first.")
            return

        def work() -> None:
            import json

            controller = self._array()
            self.events.put(("storage", f"\n$ {controller.vendor}: {label}\n"))
            result = fn(controller)
            self.events.put(("storage", json.dumps(result, indent=2, default=str) + "\n"))

        self._run_async(label, work)

    def _array_connect(self) -> None:
        """Authenticate against the array."""
        def call(controller):
            controller.connect()
            return controller.describe()

        self._array_call("connect", call)

    def _array_volumes(self) -> None:
        """List volumes."""
        self._array_call(
            "list volumes",
            lambda c: [
                {"name": v.name, "size_bytes": v.size_bytes, "serial": v.serial}
                for v in c.list_volumes()
            ],
        )

    def _array_capacity(self) -> None:
        """Report array capacity."""
        self._array_call("capacity", lambda c: c.capacity())

    def _array_snapshot(self) -> None:
        """Snapshot the volume holding the library."""
        volume = self.array_volume.get().strip()
        if not volume:
            self._notify("Enter the volume backing the shard library.")
            return
        if not messagebox.askyesno(
            "Snapshot volume",
            f"Take an array snapshot of volume '{volume}'?\n\n"
            f"This creates a point-in-time copy on the array.",
        ):
            return
        self._array_call(
            "snapshot library",
            lambda c: c.snapshot_library(volume, vault=self.session.vault),
        )

    def _refresh_library(self) -> None:
        """Reload the catalog and residency panes."""
        if self.session is None:
            return
        for row in self.catalog.get_children():
            self.catalog.delete(row)
        for entry in self.session.vault.catalog():
            self.catalog.insert("", "end", values=(
                entry["shard_id"], entry["category"], entry["location"],
                f"{entry['nbytes']:,}", entry["access_count"],
            ))
        vault = self.session.vault.stats()
        cache = self.session.cache.stats()
        resident = ", ".join(cache["resident"]) or "(nothing)"
        self._set(self.residency, (
            f"store      : {vault['total_bytes']:,} bytes across {vault['shards']} shard(s), "
            f"{len(vault['libraries'])} librar(y/ies)\n"
            f"budget     : {cache['budget_bytes']:,} bytes    "
            f"resident: {cache['current_bytes']:,}    peak: {cache['peak_bytes']:,}\n"
            f"cache      : {cache['hits']} hits / {cache['misses']} misses / "
            f"{cache['evictions']} evictions\n"
            f"paged in   : {resident}\n"
            f"memory     : {self.session.memory.stats()}\n"
        ))

    def _apply_budget(self) -> None:
        """Change the resident-set budget."""
        if self.session is None:
            return
        try:
            kb = float(self.budget_var.get())
        except ValueError:
            self._notify("Budget must be a number of kilobytes.")
            return
        self.session.cache.set_budget(int(kb * 1024))
        self._refresh_library()

    def _attach(self) -> None:
        """Attach an existing .uql library (index only)."""
        if self.session is None:
            return
        path = filedialog.askopenfilename(title="Attach shard library",
                                          filetypes=[("UltraQuant library", "*.uql")])
        if not path:
            return
        count = self.session.vault.attach(path)
        self._append(self.transcript,
                     f"[library] attached {count} shard(s) from {path} "
                     f"(index only - no payloads read)\n")
        self._refresh_library()

    def _consolidate(self) -> None:
        """Pack hot shards and snapshot."""
        if self.session is None:
            return

        def work() -> None:
            from ultraquant.interpreter.selflearn import SelfLearner

            report = SelfLearner(self.session).consolidate()
            self.events.put(("out", f"[consolidate] packed {report['packed']} shard(s)"
                                    + (f" into {report['library']}" if report["library"] else "")
                                    + (f"; snapshot {report['snapshot']}\n"
                                       if report["snapshot"] else "\n")))
            self.events.put(("refresh", None))

        self._run_async("Consolidating", work)

    def _toggle_online(self) -> None:
        """Gate network access."""
        if self.session is None:
            return
        self.session.web.set_online(self.online_var.get())
        state = "ON" if self.session.web.online else "OFF"
        self._append(self.transcript, f"[web] access {state}\n")

    def _fetch(self) -> None:
        """Fetch a URL into the contemporary stash."""
        url = self.url_var.get().strip()
        if not url:
            return
        self._submit(url)

    # ------------------------------------------------------------ LLMLS panel

    def _panel_models(self) -> list[str]:
        """Model ids currently selected, ignoring the voice header rows."""
        chosen = []
        for item in self.panel_tree.selection():
            # Having a parent is what makes a row a model; the state
            # column is empty for anything not currently loaded and is
            # not a membership test.
            if not self.panel_tree.parent(item):
                continue
            name = self.panel_tree.item(item, "text")
            if name:
                chosen.append(name)
        return chosen

    def _panel_selection_changed(self, _event=None) -> None:
        """Report how many *voices* the current selection amounts to.

        Live, because that is the number the panel actually runs on. Selecting
        four Llama derivatives must visibly say "1 voice" before the question
        is asked, not after the result comes back looking uncorroborated.
        """
        from ultraquant.interpreter.llmls import independent_groups

        chosen = self._panel_models()
        if not chosen:
            self.panel_selection.set("Select models above. Agreement counts "
                                     "by voice, not headcount.")
            return
        cards = [self._panel_cards[name] for name in chosen
                 if name in self._panel_cards]
        voices = len(independent_groups(cards)) if cards else 0
        note = ("" if voices >= 2 else
                "  -  one voice cannot corroborate itself; nothing selected "
                "here can produce evidence")
        self.panel_selection.set(
            f"{len(chosen)} model(s) selected = {voices} independent "
            f"voice(s){note}")

    def _panel_refresh(self) -> None:
        """Reload the LM Studio catalogue, grouped into independent voices."""
        def work() -> None:
            from ultraquant.interpreter.llmls import (
                LMStudioUnavailable, catalogue, independent_groups,
            )
            try:
                cards = [c for c in catalogue() if c.is_chat]
            except LMStudioUnavailable as exc:
                self.events.put(("panel_status", "LM Studio: unavailable"))
                self.events.put(("panel", f"{exc}\n"))
                self.events.put(("panel_tree", []))
                return
            groups = independent_groups(cards)
            self.events.put(("panel_status",
                             f"LM Studio: {len(cards)} chat model(s), "
                             f"{len(groups)} independent voice(s)"))
            self.events.put(("panel_tree", groups))

        self._run_async("Reading the LM Studio catalogue", work)

    def _fill_panel_tree(self, groups) -> None:
        """Render voice groups into the tree. UI thread only."""
        for row in self.panel_tree.get_children():
            self.panel_tree.delete(row)
        self._panel_cards = {}
        for index, group in enumerate(groups, 1):
            head = group[0]
            label = (f"voice {index}"
                     + (f"  ({len(group)} models = ONE source)"
                        if len(group) > 1 else ""))
            parent = self.panel_tree.insert(
                "", "end", text=label, open=True,
                values=("", head.arch or "?", head.publisher or "?"))
            for card in group:
                self._panel_cards[card.id] = card
                self.panel_tree.insert(
                    parent, "end", text=card.id,
                    values=("loaded" if card.loaded else "",
                            card.arch or "?", card.publisher or "?"))
        self._panel_selection_changed()

    def _panel_load(self, load: bool) -> None:
        """Load or unload the selected models via the ``lms`` CLI."""
        chosen = self._panel_models()
        if not chosen:
            self._notify("Select one or more models first.")
            return

        def work() -> None:
            from ultraquant.interpreter.llmls import (
                LMStudioUnavailable, TeacherPanel,
            )
            verb = "load" if load else "unload"
            try:
                panel = TeacherPanel(chosen)
            except LMStudioUnavailable as exc:
                self.events.put(("panel", f"{exc}\n"))
                return
            if not panel.cli:
                self.events.put(("panel",
                                 "The 'lms' CLI was not found, so models "
                                 "cannot be pinned or unloaded here. Asking "
                                 "still works - LM Studio loads on demand.\n"))
                return
            for name in chosen:
                ok = panel.load(name) if load else panel.unload(name)
                self.events.put(("panel",
                                 f"{verb} {name}: {'ok' if ok else 'failed'}\n"))
            self.events.put(("panel_reload", None))

        self._run_async(f"{'Loading' if load else 'Unloading'} models",
                        work)

    def _panel_ask(self) -> None:
        """Put the question to every selected model, in isolation."""
        question = self.panel_question.get().strip()
        chosen = self._panel_models()
        if not question:
            self._notify("Type a question first.")
            return
        if not chosen:
            self._notify("Select one or more models first.")
            return
        to_stash = bool(self.panel_to_stash.get())
        session = self.session

        def work() -> None:
            from ultraquant.interpreter.llmls import (
                LMStudioUnavailable, TeacherPanel,
            )
            try:
                panel = TeacherPanel(chosen)
            except LMStudioUnavailable as exc:
                self.events.put(("panel", f"{exc}\n"))
                return
            self.events.put(("panel", f"\n$ {question}\n"))
            self.events.put(("panel", panel.independence_report() + "\n"))
            try:
                if to_stash and session is not None:
                    result = panel.teach(question, session.stash)
                    consensus = result["consensus"]
                else:
                    consensus, result = panel.ask(question), None
            except LMStudioUnavailable as exc:
                self.events.put(("panel", f"{exc}\n"))
                return
            self.events.put(("panel", consensus.as_text() + "\n"))
            if result is not None:
                self.events.put(("panel",
                                 f"{result['filed']} claim(s) quarantined - "
                                 "review them on the Stash tab.\n"))
                self.events.put(("refresh", None))
            self.events.put(("panel", (
                "-> corroborated across independent voices; still quarantined "
                "until promoted.\n" if consensus.corroborated else
                "-> NOT corroborated. One voice is one source, however many "
                "models back it.\n")))

        self._run_async("Asking the panel", work)

    def _refresh_stash(self) -> None:
        """Reload the stash table."""
        if self.session is None:
            return
        for row in self.stash_view.get_children():
            self.stash_view.delete(row)
        for entry in self.session.stash.entries():
            self.stash_view.insert("", "end", values=(
                entry["id"], entry["classification"], entry["status"],
                len(entry["sources"]), entry["claim"][:160],
            ))

    def _selected_stash_id(self) -> int | None:
        """Id of the selected stash row, if any."""
        selection = self.stash_view.selection()
        if not selection:
            self._notify("Select a claim first.")
            return None
        return int(self.stash_view.item(selection[0], "values")[0])

    def _analyze(self) -> None:
        """Re-run stash classification."""
        if self.session is None:
            return
        stats = self.session.stash.analyze(self.session.memory)
        self._append(self.transcript, f"[stash] {stats}\n")
        self._refresh_stash()

    def _promote(self, force: bool = False) -> None:
        """Promote the selected claim to a stored fact."""
        from ultraquant.interpreter.stash import StashError

        entry_id = self._selected_stash_id()
        if entry_id is None or self.session is None:
            return
        try:
            key = self.session.stash.promote(entry_id, self.session.memory, force=force)
        except StashError as exc:
            self._notify(f"Not eligible: {exc}")
            return
        self.session.memory.save()
        self._append(self.transcript, f"[stash] promoted {entry_id} to fact '{key}'\n")
        self._refresh_stash()

    def _reject(self) -> None:
        """Reject the selected claim."""
        entry_id = self._selected_stash_id()
        if entry_id is None or self.session is None:
            return
        self.session.stash.reject(entry_id, "rejected in GUI")
        self._append(self.transcript, f"[stash] rejected {entry_id}\n")
        self._refresh_stash()

    def _choose_home(self) -> None:
        """Switch to a different session folder."""
        path = filedialog.askdirectory(title="Choose session folder")
        if not path:
            return
        self.home = Path(path)
        self._append(self.transcript, f"\n[session] switching to {self.home}\n")
        self._run_async("Starting session", self._start_session)

    def _about(self) -> None:
        """About box."""
        messagebox.showinfo(
            "UltraQuant",
            "UltraQuant - ultra-quantized hybrid quantum/classical pattern model.\n\n"
            "From-scratch qubit engine, ternary-weight networks, a catalogued shard\n"
            "library that pages on demand, systematic memory, the Ar(T)chive, and a\n"
            "chat interpreter that quarantines what it finds on the web.\n\n"
            "Pure Python standard library, with optional C++ and CUDA acceleration.\n"
            "See ARCHITECTURE.md.",
        )


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI.

    Args:
        argv: Optional ``[session_folder]``.

    Returns:
        Process exit code (0 on success, 1 if Tkinter is unavailable).
    """
    if tk is None:
        print("Tkinter is not available in this Python installation.", file=sys.stderr)
        print("Use the command line instead: python -m ultraquant.interpreter.chat",
              file=sys.stderr)
        return 1

    argv = list(sys.argv[1:] if argv is None else argv)
    home = Path(argv[0]) if argv else Path.cwd() / "uq_home"

    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:  # noqa: BLE001 - theming is cosmetic
        pass
    UltraQuantGUI(root, home)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
