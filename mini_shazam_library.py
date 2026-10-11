"""
Mini Shazam + Music Library - audio fingerprinting desktop app.

Install:
    pip install numpy scipy librosa matplotlib sounddevice mutagen pygame

    (mutagen = reads title/artist/album tags, pygame = built-in music player.
     Both are optional: the app still works without them, with fewer features.)

Run:
    python mini_shazam_library.py

Features:
    - Music library in SQLite: title, artist, album, duration, favorites
    - Set a music folder once; new songs are scanned automatically at startup
    - Search box + click column headers to sort + "favorites only" filter
    - Built-in player: play / pause / stop / next / volume / progress / auto-next
    - Identify a song from a clip file or the microphone (audio fingerprinting)
    - Top-3 candidates, "Play Match" button, spectrogram with detected peaks
"""

import os
import queue
import sqlite3
import hashlib
import threading
from collections import defaultdict
from datetime import datetime

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import librosa
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import pygame
    HAVE_PYGAME = True
except ImportError:
    HAVE_PYGAME = False

# ====================== settings ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "fingerprints.db")

SAMPLE_RATE = 11025            # low rate = faster, still enough for matching
N_FFT = 2048
HOP_LENGTH = 512
PEAK_NEIGHBORHOOD = (20, 20)   # (freq bins, time frames) for local-max search
MIN_AMPLITUDE_DB = -40         # ignore quiet peaks
FAN_OUT = 10                   # pair each peak with the next N peaks
MIN_DT, MAX_DT = 1, 200        # allowed time gap (frames) between paired peaks
MIN_MATCHES = 8                # min aligned hashes to accept a match
QUERY_MAX_SECONDS = 30         # only use the first 30 s of a chosen clip
AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

TITLE_SQL = "COALESCE(NULLIF(title,''), name)"


# ====================== database ======================
def get_db():
    db = sqlite3.connect(DB_FILE)
    db.execute(
        "CREATE TABLE IF NOT EXISTS songs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, "
        "path TEXT, num_hashes INTEGER, added_at TEXT)"
    )
    # migrate older databases created by the previous version of the app
    existing = {r[1] for r in db.execute("PRAGMA table_info(songs)")}
    for col, ddl in (("title", "TEXT"), ("artist", "TEXT"), ("album", "TEXT"),
                     ("duration", "REAL"), ("favorite", "INTEGER DEFAULT 0")):
        if col not in existing:
            db.execute(f"ALTER TABLE songs ADD COLUMN {col} {ddl}")
    db.execute(
        "CREATE TABLE IF NOT EXISTS fingerprints "
        "(hash TEXT, song_id INTEGER, offset INTEGER)"
    )
    db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_song ON fingerprints(song_id)")
    db.commit()
    return db


def get_setting(key, default=None):
    db = get_db()
    try:
        row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else default
    finally:
        db.close()


def set_setting(key, value):
    db = get_db()
    try:
        db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        db.commit()
    finally:
        db.close()


def get_known_paths():
    db = get_db()
    try:
        return {r[0] for r in db.execute("SELECT path FROM songs") if r[0]}
    finally:
        db.close()


def find_audio(folder):
    found = []
    for root_dir, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(AUDIO_EXT):
                found.append(os.path.abspath(os.path.join(root_dir, f)))
    return found


def fmt_time(seconds):
    seconds = int(max(0, seconds or 0))
    return f"{seconds // 60}:{seconds % 60:02d}"


# ====================== metadata ======================
def read_metadata(path):
    """Title / artist / album from tags (mutagen) or from 'Artist - Title' filenames."""
    stem = os.path.splitext(os.path.basename(path))[0]
    title, artist, album = stem, "", ""
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        artist, title = a.strip(), t.strip()
    try:
        from mutagen import File as MutagenFile
        tags = MutagenFile(path, easy=True)
        if tags:
            title = (tags.get("title") or [title])[0]
            artist = (tags.get("artist") or [artist])[0]
            album = (tags.get("album") or [album])[0]
    except Exception:
        pass
    return str(title), str(artist), str(album)


# ====================== fingerprinting core ======================
def load_audio(path, duration=None):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=duration)
    return y


def find_peaks(y):
    """Return (peaks, spec_db). peaks = [(time_frame, freq_bin), ...] sorted by time."""
    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    local_max = maximum_filter(spec_db, size=PEAK_NEIGHBORHOOD) == spec_db
    mask = local_max & (spec_db > MIN_AMPLITUDE_DB)

    freq_idx, time_idx = np.where(mask)
    peaks = sorted(zip(time_idx.tolist(), freq_idx.tolist()))
    return peaks, spec_db


def generate_hashes(peaks):
    """Pair nearby peaks -> list of (hash, time_frame_of_first_peak)."""
    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        for t2, f2 in peaks[i + 1: i + 1 + FAN_OUT]:
            dt = t2 - t1
            if MIN_DT <= dt <= MAX_DT:
                key = f"{f1}|{f2}|{dt}".encode()
                hashes.append((hashlib.sha1(key).hexdigest()[:20], t1))
    return hashes


def add_song(path):
    """Fingerprint a song file and store it. Returns 'added' or 'skipped'."""
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    db = get_db()
    try:
        if db.execute("SELECT 1 FROM songs WHERE path=? OR name=?", (path, name)).fetchone():
            return "skipped"

        y = load_audio(path)
        duration = len(y) / SAMPLE_RATE
        peaks, _ = find_peaks(y)
        hashes = generate_hashes(peaks)
        title, artist, album = read_metadata(path)

        cur = db.execute(
            "INSERT INTO songs (name, path, num_hashes, added_at, title, artist, album, duration, favorite) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (name, path, len(hashes), datetime.now().strftime("%Y-%m-%d %H:%M"),
             title, artist, album, duration),
        )
        song_id = cur.lastrowid
        db.executemany(
            "INSERT INTO fingerprints (hash, song_id, offset) VALUES (?, ?, ?)",
            [(h, song_id, off) for h, off in hashes],
        )
        db.commit()
        return "added"
    finally:
        db.close()


def identify_audio(y):
    """
    Match audio samples against the library.
    Returns dict: matches [(song_id, label, score, pct)], peaks, spec_db,
    n_hashes, library_size.
    """
    peaks, spec_db = find_peaks(y)
    q_hashes = generate_hashes(peaks)

    out = {"matches": [], "peaks": peaks, "spec_db": spec_db,
           "n_hashes": len(q_hashes), "library_size": 0}

    db = get_db()
    try:
        out["library_size"] = db.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        if not q_hashes or out["library_size"] == 0:
            return out

        q_map = defaultdict(list)          # hash -> offsets in the query
        for h, off in q_hashes:
            q_map[h].append(off)
        unique = list(q_map.keys())

        # vote for (song, time-offset difference); the real song gets a spike
        votes = defaultdict(int)
        for i in range(0, len(unique), 500):
            chunk = unique[i:i + 500]
            marks = ",".join("?" * len(chunk))
            rows = db.execute(
                f"SELECT hash, song_id, offset FROM fingerprints WHERE hash IN ({marks})",
                chunk,
            ).fetchall()
            for h, song_id, db_off in rows:
                for q_off in q_map[h]:
                    votes[(song_id, db_off - q_off)] += 1

        best_per_song = {}
        for (song_id, _diff), count in votes.items():
            if count > best_per_song.get(song_id, 0):
                best_per_song[song_id] = count

        ranked = sorted(best_per_song.items(), key=lambda kv: -kv[1])[:3]
        for song_id, score in ranked:
            row = db.execute(
                f"SELECT {TITLE_SQL}, COALESCE(artist,'') FROM songs WHERE id=?", (song_id,)
            ).fetchone()
            label = row[0] + (f" - {row[1]}" if row[1] else "")
            pct = 100.0 * score / max(1, len(q_hashes))
            out["matches"].append((song_id, label, score, pct))
        return out
    finally:
        db.close()


def record_from_mic(seconds):
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("The 'sounddevice' package is missing.\n\nInstall it with:\n    pip install sounddevice")
    rec = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                 channels=1, dtype="float32")
    sd.wait()
    return rec.flatten()


# ====================== GUI ======================
class App:
    BG = "#10131a"
    PANEL = "#181d27"
    FIELD = "#232a3a"
    FG = "#e8ecf3"
    MUTED = "#8b93a5"
    ACCENT = "#4f8cff"
    GOOD = "#3ddc97"
    BAD = "#ff6b6b"
    GOLD = "#ffd166"

    SORT_SQL = {
        "fav": "COALESCE(favorite,0)",
        "title": f"{TITLE_SQL} COLLATE NOCASE",
        "artist": "COALESCE(artist,'') COLLATE NOCASE",
        "album": "COALESCE(album,'') COLLATE NOCASE",
        "dur": "COALESCE(duration,0)",
        "prints": "num_hashes",
    }

    def __init__(self, root):
        self.root = root
        root.title("Mini Shazam - Music Library")
        root.geometry("1220x860")
        root.minsize(1050, 740)
        root.configure(bg=self.BG)

        self.q = queue.Queue()
        self.busy = False
        self.now = None          # currently playing song dict
        self.paused = False
        self.sort_col, self.sort_desc = "title", False

        self._setup_style()
        self._build_ui()
        self.refresh_library()

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self.poll)
        root.after(500, self.tick)
        root.after(700, self.auto_scan)

    # ---------- styling ----------
    def _setup_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=self.BG, foreground=self.FG, fieldbackground=self.PANEL)
        s.configure("TFrame", background=self.BG)
        s.configure("Panel.TLabelframe", background=self.PANEL, foreground=self.FG,
                    bordercolor="#262c3a")
        s.configure("Panel.TLabelframe.Label", background=self.PANEL, foreground=self.ACCENT,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TLabel", background=self.PANEL, foreground=self.FG)
        s.configure("TButton", padding=6, background=self.FIELD, foreground=self.FG,
                    bordercolor="#2d3548", focuscolor=self.FIELD)
        s.map("TButton", background=[("active", "#2d3750")])
        s.configure("Accent.TButton", background=self.ACCENT, foreground="white")
        s.map("Accent.TButton", background=[("active", "#3a73d8")])
        s.configure("Treeview", background=self.PANEL, foreground=self.FG,
                    fieldbackground=self.PANEL, rowheight=26, bordercolor="#262c3a")
        s.configure("Treeview.Heading", background=self.FIELD, foreground=self.FG,
                    font=("Segoe UI", 9, "bold"))
        s.map("Treeview", background=[("selected", "#2b4a86")])
        s.configure("TSpinbox", fieldbackground=self.FIELD, foreground=self.FG)
        s.configure("TEntry", fieldbackground=self.FIELD, foreground=self.FG,
                    insertcolor=self.FG)
        s.configure("TCheckbutton", background=self.PANEL, foreground=self.FG)
        s.map("TCheckbutton", background=[("active", self.PANEL)])
        s.configure("Horizontal.TProgressbar", background=self.ACCENT, troughcolor=self.FIELD)
        s.configure("Horizontal.TScale", background=self.PANEL, troughcolor=self.FIELD)

    # ---------- layout ----------
    def _build_ui(self):
        tk.Label(self.root, text="Mini Shazam  +  Music Library", font=("Segoe UI", 20, "bold"),
                 bg=self.BG, fg=self.FG, anchor="w").pack(fill="x", padx=18, pady=(12, 0))
        tk.Label(self.root,
                 text="Browse and play your music, and identify songs with audio fingerprinting - no internet needed",
                 font=("Segoe UI", 10), bg=self.BG, fg=self.MUTED, anchor="w"
                 ).pack(fill="x", padx=18, pady=(0, 6))

        # bottom bars are packed first so they always stay visible
        status_bar = tk.Frame(self.root, bg=self.BG)
        status_bar.pack(side="bottom", fill="x", padx=16, pady=(2, 8))
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(status_bar, textvariable=self.status_var, bg=self.BG, fg=self.MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=180)
        self.progress.pack(side="right")

        self._build_player()

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=14)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=3)
        body.rowconfigure(1, weight=2)

        self._build_library(body)
        self._build_identify(body)

        plot_box = ttk.LabelFrame(body, text="  Spectrogram and Detected Peaks (clip analysed)  ",
                                  style="Panel.TLabelframe")
        plot_box.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        plot_box.columnconfigure(0, weight=1)
        plot_box.rowconfigure(0, weight=1)
        self.fig = Figure(figsize=(8, 2.4), dpi=100, facecolor=self.PANEL)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._plot_placeholder()

    def _build_library(self, body):
        box = ttk.LabelFrame(body, text="  Music Library  ", style="Panel.TLabelframe")
        box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(1, weight=1)

        # search + filter
        top = tk.Frame(box, bg=self.PANEL)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        tk.Label(top, text="Search:", bg=self.PANEL, fg=self.MUTED).pack(side="left")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self.refresh_library())
        ttk.Entry(top, textvariable=self.search_var, width=28).pack(side="left", padx=6)
        self.fav_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Favorites only", variable=self.fav_only,
                        command=self.refresh_library).pack(side="left", padx=10)

        # table
        cols = ("fav", "title", "artist", "album", "dur", "prints")
        self.lib = ttk.Treeview(box, columns=cols, show="headings", selectmode="extended")
        spec = (("fav", "*", 30, "center"), ("title", "Title", 190, "w"),
                ("artist", "Artist", 130, "w"), ("album", "Album", 120, "w"),
                ("dur", "Time", 55, "center"), ("prints", "Prints", 70, "e"))
        for c, t, w, a in spec:
            self.lib.heading(c, text=t, command=lambda c=c: self.sort_by(c))
            self.lib.column(c, width=w, anchor=a)
        self.lib.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=8)
        sb = ttk.Scrollbar(box, orient="vertical", command=self.lib.yview)
        sb.grid(row=1, column=1, sticky="ns", pady=8, padx=(0, 8))
        self.lib.configure(yscrollcommand=sb.set)
        self.lib.bind("<Double-1>", self.on_lib_double_click)

        # buttons (two rows)
        b1 = tk.Frame(box, bg=self.PANEL)
        b1.grid(row=2, column=0, columnspan=2, sticky="w", padx=8)
        ttk.Button(b1, text="Add Songs", command=self.add_files).pack(side="left", padx=(0, 6))
        ttk.Button(b1, text="Add Folder", command=self.add_folder).pack(side="left", padx=(0, 6))
        ttk.Button(b1, text="Set Music Folder", command=self.set_music_folder).pack(side="left", padx=(0, 6))
        ttk.Button(b1, text="Rescan", command=self.rescan).pack(side="left")

        b2 = tk.Frame(box, bg=self.PANEL)
        b2.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 0))
        ttk.Button(b2, text="Toggle Favorite", command=self.toggle_favorite).pack(side="left", padx=(0, 6))
        ttk.Button(b2, text="Remove Selected", command=self.remove_selected).pack(side="left")

        self.stats_var = tk.StringVar(value="")
        tk.Label(box, textvariable=self.stats_var, bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9), anchor="w", justify="left"
                 ).grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(6, 8))

    def _build_identify(self, body):
        idf = ttk.LabelFrame(body, text="  Identify a Song  ", style="Panel.TLabelframe")
        idf.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        idf.columnconfigure(0, weight=1)

        ctrl = tk.Frame(idf, bg=self.PANEL)
        ctrl.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        ttk.Button(ctrl, text="Choose Clip...", command=self.identify_file).pack(side="left")
        ttk.Button(ctrl, text="Listen (Mic)", style="Accent.TButton",
                   command=self.identify_mic).pack(side="left", padx=8)
        tk.Label(ctrl, text="for", bg=self.PANEL, fg=self.MUTED).pack(side="left")
        self.secs = tk.IntVar(value=8)
        ttk.Spinbox(ctrl, from_=3, to=20, width=4, textvariable=self.secs).pack(side="left", padx=4)
        tk.Label(ctrl, text="sec", bg=self.PANEL, fg=self.MUTED).pack(side="left")

        self.result_var = tk.StringVar(value="Add some songs, then identify a clip")
        self.result_lbl = tk.Label(idf, textvariable=self.result_var, bg=self.PANEL, fg=self.FG,
                                   font=("Segoe UI", 14, "bold"), wraplength=400,
                                   justify="left", anchor="w")
        self.result_lbl.grid(row=1, column=0, sticky="ew", padx=12, pady=(8, 2))

        self.detail_var = tk.StringVar(value="")
        tk.Label(idf, textvariable=self.detail_var, bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9), anchor="w", justify="left", wraplength=400
                 ).grid(row=2, column=0, sticky="ew", padx=12)

        rcols = ("rank", "song", "score", "pct")
        self.res = ttk.Treeview(idf, columns=rcols, show="headings", height=3, selectmode="browse")
        for c, t, w, a in (("rank", "#", 30, "center"), ("song", "Candidate", 210, "w"),
                           ("score", "Score", 55, "e"), ("pct", "Aligned", 65, "e")):
            self.res.heading(c, text=t)
            self.res.column(c, width=w, anchor=a)
        self.res.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 6))
        self.res.bind("<Double-1>", lambda e: self.play_match())

        ttk.Button(idf, text="Play Selected Match", command=self.play_match
                   ).grid(row=4, column=0, sticky="w", padx=12, pady=(0, 10))

    def _build_player(self):
        box = ttk.LabelFrame(self.root, text="  Now Playing  ", style="Panel.TLabelframe")
        box.pack(side="bottom", fill="x", padx=18, pady=(4, 4))
        box.columnconfigure(1, weight=1)

        btns = tk.Frame(box, bg=self.PANEL)
        btns.grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Button(btns, text="Play", style="Accent.TButton", width=7,
                   command=self.play_selected).pack(side="left", padx=(0, 6))
        self.pause_btn = ttk.Button(btns, text="Pause", width=8, command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Stop", width=6, command=self.stop_song).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Next", width=6, command=self.next_song).pack(side="left")

        mid = tk.Frame(box, bg=self.PANEL)
        mid.grid(row=0, column=1, sticky="ew", padx=10)
        mid.columnconfigure(0, weight=1)
        self.now_var = tk.StringVar(value="Nothing playing")
        tk.Label(mid, textvariable=self.now_var, bg=self.PANEL, fg=self.FG,
                 font=("Segoe UI", 10, "bold"), anchor="w").grid(row=0, column=0, sticky="ew")
        self.play_prog = ttk.Progressbar(mid, mode="determinate", maximum=100)
        self.play_prog.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.time_var = tk.StringVar(value="0:00 / 0:00")
        tk.Label(box, textvariable=self.time_var, bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9)).grid(row=0, column=2, padx=8)

        vol = tk.Frame(box, bg=self.PANEL)
        vol.grid(row=0, column=3, padx=(4, 12))
        tk.Label(vol, text="Vol", bg=self.PANEL, fg=self.MUTED).pack(side="left", padx=(0, 4))
        self.vol = tk.DoubleVar(value=70)
        ttk.Scale(vol, from_=0, to=100, variable=self.vol, length=110,
                  command=self.on_volume).pack(side="left")

    # ---------- plotting ----------
    def _style_axes(self):
        self.ax.set_facecolor("#0b0e14")
        self.ax.tick_params(colors=self.MUTED, labelsize=8)
        for sp in self.ax.spines.values():
            sp.set_color("#262c3a")
        self.ax.set_xlabel("Time (s)", color=self.MUTED, fontsize=9)
        self.ax.set_ylabel("Frequency (Hz)", color=self.MUTED, fontsize=9)

    def _plot_placeholder(self):
        self.ax.clear()
        self._style_axes()
        self.ax.text(0.5, 0.5, "Identify a clip to see its spectrogram and fingerprint peaks",
                     ha="center", va="center", color=self.MUTED, fontsize=11,
                     transform=self.ax.transAxes)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_spectrogram(self, spec_db, peaks):
        self.ax.clear()
        self._style_axes()
        t_max = spec_db.shape[1] * HOP_LENGTH / SAMPLE_RATE
        self.ax.imshow(spec_db, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, t_max, 0, SAMPLE_RATE / 2], vmin=-80, vmax=0)
        if peaks:
            pts = peaks
            if len(pts) > 4000:  # keep drawing fast
                idx = np.random.choice(len(pts), 4000, replace=False)
                pts = [peaks[i] for i in idx]
            xs = [t * HOP_LENGTH / SAMPLE_RATE for t, _ in pts]
            ys = [f * SAMPLE_RATE / N_FFT for _, f in pts]
            self.ax.scatter(xs, ys, s=7, c="#3ddc97", marker="o", linewidths=0, alpha=0.9)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---------- background tasks (keep UI responsive) ----------
    def set_status(self, text, busy=None):
        self.status_var.set(text)
        if busy is True:
            self.progress.start(12)
        elif busy is False:
            self.progress.stop()

    def run_task(self, func, on_done, status):
        if self.busy:
            messagebox.showinfo("Please wait", "Another task is still running.")
            return
        self.busy = True
        self.set_status(status, True)

        def worker():
            try:
                self.q.put(("ok", on_done, func()))
            except Exception as e:
                self.q.put(("err", None, e))

        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, cb, payload = self.q.get_nowait()
                if kind == "progress":
                    self.set_status(payload)
                elif kind == "ok":
                    self.busy = False
                    self.set_status("Ready", False)
                    cb(payload)
                else:
                    self.busy = False
                    self.set_status("Ready", False)
                    messagebox.showerror("Error", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    # ---------- library ----------
    def sort_by(self, col):
        if self.sort_col == col:
            self.sort_desc = not self.sort_desc
        else:
            self.sort_col, self.sort_desc = col, (col in ("fav", "dur", "prints"))
        self.refresh_library()

    def refresh_library(self):
        for item in self.lib.get_children():
            self.lib.delete(item)

        term = f"%{self.search_var.get().strip()}%"
        where = f"({TITLE_SQL} LIKE ? OR COALESCE(artist,'') LIKE ? OR COALESCE(album,'') LIKE ?)"
        if self.fav_only.get():
            where += " AND COALESCE(favorite,0)=1"
        order = self.SORT_SQL[self.sort_col] + (" DESC" if self.sort_desc else " ASC")

        db = get_db()
        try:
            rows = db.execute(
                f"SELECT id, COALESCE(favorite,0), {TITLE_SQL}, COALESCE(artist,''), "
                f"COALESCE(album,''), COALESCE(duration,0), num_hashes FROM songs "
                f"WHERE {where} ORDER BY {order}", (term, term, term)
            ).fetchall()
            total, prints = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(num_hashes),0) FROM songs").fetchone()
        finally:
            db.close()

        for sid, fav, title, artist, album, dur, n in rows:
            self.lib.insert("", "end", iid=str(sid),
                            values=("\u2605" if fav else "", title, artist, album,
                                    fmt_time(dur), f"{n:,}"))
        folder = get_setting("music_folder")
        self.stats_var.set(
            f"Showing {len(rows)} of {total} songs  |  {prints:,} fingerprints stored\n"
            f"Music folder: {folder if folder else '(not set)'}"
        )

    def on_lib_double_click(self, event):
        row = self.lib.identify_row(event.y)
        if row:
            self.play_song(int(row))

    # ----- adding songs -----
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select songs",
            filetypes=[("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"), ("All files", "*.*")],
        )
        if paths:
            self.start_adding(list(paths))

    def add_folder(self):
        folder = filedialog.askdirectory(title="Add all songs from a folder")
        if folder:
            self.start_scan(folder)

    def set_music_folder(self):
        folder = filedialog.askdirectory(title="Choose your main music folder")
        if folder:
            set_setting("music_folder", os.path.abspath(folder))
            self.start_scan(folder)

    def rescan(self):
        folder = get_setting("music_folder")
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo("No music folder",
                                "Click 'Set Music Folder' first to choose where your songs live.")
            return
        self.start_scan(folder)

    def auto_scan(self):
        folder = get_setting("music_folder")
        if folder and os.path.isdir(folder):
            self.start_scan(folder, quiet=True)

    def _add_job(self, get_paths):
        def job():
            paths = get_paths()
            added = skipped = failed = 0
            for i, p in enumerate(paths, 1):
                self.q.put(("progress", None,
                            f"Fingerprinting {i}/{len(paths)}: {os.path.basename(p)}"))
                try:
                    if add_song(p) == "added":
                        added += 1
                    else:
                        skipped += 1
                except Exception:
                    failed += 1
            return added, skipped, failed, len(paths)
        return job

    def start_adding(self, paths):
        self.run_task(self._add_job(lambda: paths),
                      lambda r: self.after_add(r), "Adding songs...")

    def start_scan(self, folder, quiet=False):
        def get_paths():
            known = get_known_paths()
            return [p for p in find_audio(folder) if p not in known]
        self.run_task(self._add_job(get_paths),
                      lambda r: self.after_add(r, quiet), "Scanning music folder...")

    def after_add(self, res, quiet=False):
        added, skipped, failed, found = res
        self.refresh_library()
        if quiet and found == 0:
            self.set_status("Library is up to date")
            return
        msg = f"Added {added} new song(s), skipped {skipped}"
        if failed:
            msg += f", failed {failed}"
        self.set_status(msg)

    # ----- editing -----
    def toggle_favorite(self):
        sel = self.lib.selection()
        if not sel:
            messagebox.showinfo("Nothing selected", "Select one or more songs first.")
            return
        db = get_db()
        try:
            for sid in sel:
                db.execute("UPDATE songs SET favorite = 1 - COALESCE(favorite,0) WHERE id=?",
                           (int(sid),))
            db.commit()
        finally:
            db.close()
        self.refresh_library()

    def remove_selected(self):
        if self.busy:
            return
        sel = self.lib.selection()
        if not sel:
            messagebox.showinfo("Nothing selected", "Select one or more songs to remove.")
            return
        if not messagebox.askyesno(
                "Remove songs",
                f"Remove {len(sel)} song(s) from the library?\n(The audio files stay on your disk.)"):
            return
        if self.now and str(self.now["id"]) in sel:
            self.stop_song()
        db = get_db()
        try:
            for sid in sel:
                db.execute("DELETE FROM fingerprints WHERE song_id=?", (int(sid),))
                db.execute("DELETE FROM songs WHERE id=?", (int(sid),))
            db.commit()
        finally:
            db.close()
        self.refresh_library()
        self.set_status(f"Removed {len(sel)} song(s)")

    # ---------- player ----------
    def _need_pygame(self):
        if not HAVE_PYGAME:
            messagebox.showinfo("Player unavailable",
                                "Install pygame to enable the built-in player:\n\n    pip install pygame")
            return True
        return False

    def play_song(self, song_id):
        if self._need_pygame():
            return
        db = get_db()
        try:
            row = db.execute(
                f"SELECT path, {TITLE_SQL}, COALESCE(artist,''), COALESCE(duration,0) "
                f"FROM songs WHERE id=?", (song_id,)).fetchone()
        finally:
            db.close()
        if not row:
            return
        path, title, artist, dur = row
        if not path or not os.path.isfile(path):
            messagebox.showerror("File missing",
                                 f"The audio file could not be found:\n{path}")
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(self.vol.get() / 100.0)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Cannot play this file", str(e))
            return

        self.now = {"id": song_id, "duration": dur}
        self.paused = False
        self.pause_btn.configure(text="Pause")
        self.now_var.set(title + (f" - {artist}" if artist else ""))
        if self.lib.exists(str(song_id)):
            self.lib.selection_set(str(song_id))
            self.lib.see(str(song_id))

    def _selected_lib_id(self):
        sel = self.lib.selection()
        if sel:
            return int(sel[0])
        items = self.lib.get_children()
        return int(items[0]) if items else None

    def play_selected(self):
        if self._need_pygame():
            return
        if self.paused and self.now and not self.lib.selection():
            self.toggle_pause()
            return
        sid = self._selected_lib_id()
        if sid is None:
            messagebox.showinfo("Empty library", "Add some songs first.")
            return
        self.play_song(sid)

    def play_match(self):
        sel = self.res.selection() or self.res.get_children()[:1]
        if not sel:
            messagebox.showinfo("No match yet", "Identify a clip first.")
            return
        self.play_song(int(sel[0]))

    def toggle_pause(self):
        if not HAVE_PYGAME or not self.now:
            return
        if self.paused:
            pygame.mixer.music.unpause()
            self.paused = False
            self.pause_btn.configure(text="Pause")
        else:
            pygame.mixer.music.pause()
            self.paused = True
            self.pause_btn.configure(text="Resume")

    def stop_song(self):
        if HAVE_PYGAME and pygame.mixer.get_init():
            pygame.mixer.music.stop()
        self._reset_player_ui()

    def _reset_player_ui(self):
        self.now = None
        self.paused = False
        self.pause_btn.configure(text="Pause")
        self.now_var.set("Nothing playing")
        self.play_prog["value"] = 0
        self.time_var.set("0:00 / 0:00")

    def next_song(self):
        if self._need_pygame():
            return
        items = list(self.lib.get_children())
        if not items:
            return
        if self.now and str(self.now["id"]) in items:
            idx = items.index(str(self.now["id"]))
            if idx + 1 < len(items):
                self.play_song(int(items[idx + 1]))
            else:
                self.stop_song()
        else:
            self.play_selected()

    def on_volume(self, _value=None):
        if HAVE_PYGAME and pygame.mixer.get_init():
            pygame.mixer.music.set_volume(self.vol.get() / 100.0)

    def tick(self):
        """Updates the progress bar and moves to the next song when one ends."""
        if HAVE_PYGAME and self.now and pygame.mixer.get_init():
            busy = pygame.mixer.music.get_busy()
            if busy or self.paused:
                pos = max(0, pygame.mixer.music.get_pos()) / 1000.0
                dur = self.now["duration"] or 0
                self.play_prog["value"] = min(100, 100.0 * pos / dur) if dur else 0
                self.time_var.set(f"{fmt_time(pos)} / {fmt_time(dur)}")
            else:
                finished = self.now["id"]
                items = list(self.lib.get_children())
                self.now = None
                if str(finished) in items:
                    idx = items.index(str(finished))
                    if idx + 1 < len(items):
                        self.play_song(int(items[idx + 1]))   # auto-next
                if self.now is None:
                    self._reset_player_ui()
        self.root.after(500, self.tick)

    def on_close(self):
        try:
            if HAVE_PYGAME and pygame.mixer.get_init():
                pygame.mixer.quit()
        except Exception:
            pass
        self.root.destroy()

    # ---------- identify ----------
    def identify_file(self):
        path = filedialog.askopenfilename(
            title="Choose an audio clip",
            filetypes=[("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"), ("All files", "*.*")],
        )
        if not path:
            return
        self.run_task(
            lambda: identify_audio(load_audio(path, duration=QUERY_MAX_SECONDS)),
            self.show_result,
            f"Analysing {os.path.basename(path)}...",
        )

    def identify_mic(self):
        try:
            seconds = max(3, min(20, int(self.secs.get())))
        except (tk.TclError, ValueError):
            seconds = 8
        if self.now and not self.paused:
            self.toggle_pause()   # don't let the app's own player confuse the mic
        self.run_task(
            lambda: identify_audio(record_from_mic(seconds)),
            self.show_result,
            f"Listening for {seconds} seconds - play the song now...",
        )

    def show_result(self, res):
        for item in self.res.get_children():
            self.res.delete(item)

        self.plot_spectrogram(res["spec_db"], res["peaks"])

        if res["library_size"] == 0:
            self.result_var.set("Your library is empty")
            self.result_lbl.configure(fg=self.BAD)
            self.detail_var.set("Add songs with 'Add Songs' or 'Set Music Folder' first.")
            return

        if res["n_hashes"] == 0:
            self.result_var.set("Could not analyse the audio")
            self.result_lbl.configure(fg=self.BAD)
            self.detail_var.set("The clip may be silent or too short. Try a louder or longer clip.")
            return

        matches = res["matches"]
        for rank, (sid, label, score, pct) in enumerate(matches, 1):
            self.res.insert("", "end", iid=str(sid),
                            values=(rank, label, score, f"{pct:.1f}%"))
        if matches:
            self.res.selection_set(str(matches[0][0]))

        if matches and matches[0][2] >= MIN_MATCHES:
            self.result_var.set(f"Match: {matches[0][1]}")
            self.result_lbl.configure(fg=self.GOOD)
            self.detail_var.set(
                f"{matches[0][2]} fingerprints lined up at the same time offset "
                f"(clip produced {res['n_hashes']} fingerprints)."
            )
        else:
            self.result_var.set("No confident match found")
            self.result_lbl.configure(fg=self.BAD)
            self.detail_var.set(
                "The song may not be in your library, or the clip is too noisy/short. "
                "Try a longer clip."
            )
        self.set_status("Done")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
