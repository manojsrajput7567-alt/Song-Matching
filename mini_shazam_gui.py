import os
import queue
import sqlite3
import hashlib
import threading
from collections import defaultdict
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import librosa
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
AUDIO_PATTERNS = " ".join(f"*{e}" for e in AUDIO_EXT)  # for file dialogs
MIN_MATCH_PCT = 1.5                            # also require this share of query hashes


# ====================== database ======================
def get_db():
    db = sqlite3.connect(DB_FILE)
    db.execute(
        "CREATE TABLE IF NOT EXISTS songs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, "
        "path TEXT, num_hashes INTEGER, added_at TEXT)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS fingerprints "
        "(hash TEXT, song_id INTEGER, offset INTEGER)"
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_song ON fingerprints(song_id)")
    db.commit()
    return db


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


def unique_name(db, base):
    """Disambiguate same-named songs from different folders (name is UNIQUE)."""
    name = base
    n = 1
    while db.execute("SELECT 1 FROM songs WHERE name=?", (name,)).fetchone():
        n += 1
        name = f"{base} ({n})"
    return name


def add_song(path):
    """Fingerprint a song file and store it. Returns 'added' or 'skipped'."""
    base = os.path.splitext(os.path.basename(path))[0]
    db = get_db()
    try:
        # same exact file already fingerprinted?
        if db.execute("SELECT 1 FROM songs WHERE path=?", (path,)).fetchone():
            return "skipped"
        name = unique_name(db, base)

        y = load_audio(path)
        peaks, _ = find_peaks(y)
        hashes = generate_hashes(peaks)

        cur = db.execute(
            "INSERT INTO songs (name, path, num_hashes, added_at) VALUES (?, ?, ?, ?)",
            (name, path, len(hashes), datetime.now().strftime("%Y-%m-%d %H:%M")),
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
    Returns dict with: matches [(name, score, pct)], peaks, spec_db,
    n_hashes, library_size.
    """
    peaks, spec_db = find_peaks(y)
    q_hashes = generate_hashes(peaks)

    out = {
        "matches": [],
        "peaks": peaks,
        "spec_db": spec_db,
        "n_hashes": len(q_hashes),
        "library_size": 0,
    }

    db = get_db()
    try:
        out["library_size"] = db.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        if not q_hashes or out["library_size"] == 0:
            return out

        # hash -> list of offsets where it occurs in the query
        q_map = defaultdict(list)
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
        if not ranked:
            return out

        # one query for all candidate names, not one per candidate
        ids = [sid for sid, _ in ranked]
        marks = ",".join("?" * len(ids))
        names = dict(db.execute(
            f"SELECT id, name FROM songs WHERE id IN ({marks})", ids
        ).fetchall())

        for song_id, score in ranked:
            # divide by UNIQUE query hashes, else duplicate hashes inflate pct
            pct = 100.0 * score / len(unique)
            if score < MIN_MATCHES or pct < MIN_MATCH_PCT:
                continue
            out["matches"].append((names.get(song_id, "?"), score, pct))
        return out
    finally:
        db.close()


def record_from_mic(seconds):
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("The 'sounddevice' package is missing.\n\nInstall it with:\n    pip install sounddevice")
    # Windows/WASAPI devices often refuse odd rates like 11025 - fall back
    try:
        rec = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                     channels=1, dtype="float32")
        sd.wait()
    except sd.PortAudioError:
        fallback = 44100
        rec = sd.rec(int(seconds * fallback), samplerate=fallback,
                     channels=1, dtype="float32")
        sd.wait()
        rec = librosa.resample(rec.flatten(), orig_sr=fallback, target_sr=SAMPLE_RATE)
        return rec
    return rec.flatten()


# ====================== GUI ======================
class App:
    BG = "#10131a"
    PANEL = "#181d27"
    FG = "#e8ecf3"
    MUTED = "#8b93a5"
    ACCENT = "#4f8cff"
    GOOD = "#3ddc97"
    BAD = "#ff6b6b"

    def __init__(self, root):
        self.root = root
        root.title("Mini Shazam - Audio Fingerprinting")
        root.geometry("1120x780")
        root.minsize(980, 680)
        root.configure(bg=self.BG)

        self.q = queue.Queue()
        self.busy = False

        self._setup_style()
        self._build_ui()
        self.refresh_library()
        self.root.after(100, self.poll)

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
        s.configure("TButton", padding=7, background="#232a3a", foreground=self.FG,
                    bordercolor="#2d3548", focuscolor="#232a3a")
        s.map("TButton", background=[("active", "#2d3750")])
        s.configure("Accent.TButton", background=self.ACCENT, foreground="white")
        s.map("Accent.TButton", background=[("active", "#3a73d8")])
        s.configure("Treeview", background=self.PANEL, foreground=self.FG,
                    fieldbackground=self.PANEL, rowheight=26, bordercolor="#262c3a")
        s.configure("Treeview.Heading", background="#232a3a", foreground=self.FG,
                    font=("Segoe UI", 9, "bold"))
        s.map("Treeview", background=[("selected", "#2b4a86")])
        s.configure("TSpinbox", fieldbackground="#232a3a", foreground=self.FG)
        s.configure("Horizontal.TProgressbar", background=self.ACCENT, troughcolor=self.PANEL)

    # ---------- layout ----------
    def _build_ui(self):
        header = tk.Label(self.root, text="Mini Shazam", font=("Segoe UI", 20, "bold"),
                          bg=self.BG, fg=self.FG, anchor="w")
        header.pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(self.root, text="Identify songs using audio fingerprinting - no internet or API needed",
                 font=("Segoe UI", 10), bg=self.BG, fg=self.MUTED, anchor="w"
                 ).pack(fill="x", padx=18, pady=(0, 8))

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=14)
        body.columnconfigure(0, weight=1, uniform="col")
        body.columnconfigure(1, weight=1, uniform="col")
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        # ----- library panel -----
        lib_box = ttk.LabelFrame(body, text="  Song Library (Database)  ", style="Panel.TLabelframe")
        lib_box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        lib_box.columnconfigure(0, weight=1)
        lib_box.rowconfigure(0, weight=1)

        cols = ("name", "prints", "added")
        self.lib = ttk.Treeview(lib_box, columns=cols, show="headings", selectmode="extended")
        self.lib.heading("name", text="Song")
        self.lib.heading("prints", text="Fingerprints")
        self.lib.heading("added", text="Added")
        self.lib.column("name", width=230)
        self.lib.column("prints", width=90, anchor="e")
        self.lib.column("added", width=120, anchor="center")
        self.lib.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        sb = ttk.Scrollbar(lib_box, orient="vertical", command=self.lib.yview)
        sb.grid(row=0, column=1, sticky="ns", pady=8, padx=(0, 8))
        self.lib.configure(yscrollcommand=sb.set)

        btns = tk.Frame(lib_box, bg=self.PANEL)
        btns.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 6))
        ttk.Button(btns, text="Add Songs", command=self.add_files).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Add Folder", command=self.add_folder).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Remove Selected", command=self.remove_selected).pack(side="left")

        self.stats_var = tk.StringVar(value="")
        tk.Label(lib_box, textvariable=self.stats_var, bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9)).grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))

        # ----- identify panel -----
        idf = ttk.LabelFrame(body, text="  Identify a Song  ", style="Panel.TLabelframe")
        idf.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        idf.columnconfigure(0, weight=1)

        ctrl = tk.Frame(idf, bg=self.PANEL)
        ctrl.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        ttk.Button(ctrl, text="Choose Clip File...", command=self.identify_file).pack(side="left")
        ttk.Button(ctrl, text="Listen (Microphone)", style="Accent.TButton",
                   command=self.identify_mic).pack(side="left", padx=8)
        tk.Label(ctrl, text="for", bg=self.PANEL, fg=self.MUTED).pack(side="left")
        self.secs = tk.IntVar(value=8)
        ttk.Spinbox(ctrl, from_=3, to=20, width=4, textvariable=self.secs).pack(side="left", padx=4)
        tk.Label(ctrl, text="sec", bg=self.PANEL, fg=self.MUTED).pack(side="left")

        self.result_var = tk.StringVar(value="Add some songs, then identify a clip")
        self.result_lbl = tk.Label(idf, textvariable=self.result_var, bg=self.PANEL, fg=self.FG,
                                   font=("Segoe UI", 15, "bold"), wraplength=460,
                                   justify="left", anchor="w")
        self.result_lbl.grid(row=1, column=0, sticky="ew", padx=12, pady=(10, 4))

        self.detail_var = tk.StringVar(value="")
        tk.Label(idf, textvariable=self.detail_var, bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9), anchor="w", justify="left", wraplength=460
                 ).grid(row=2, column=0, sticky="ew", padx=12)

        rcols = ("rank", "song", "score", "pct")
        self.res = ttk.Treeview(idf, columns=rcols, show="headings", height=3)
        for c, t, w, a in (("rank", "#", 35, "center"), ("song", "Candidate", 240, "w"),
                           ("score", "Score", 70, "e"), ("pct", "Aligned %", 80, "e")):
            self.res.heading(c, text=t)
            self.res.column(c, width=w, anchor=a)
        self.res.grid(row=3, column=0, sticky="ew", padx=12, pady=(10, 12))

        # ----- spectrogram panel -----
        plot_box = ttk.LabelFrame(body, text="  Spectrogram and Detected Peaks (clip analysed)  ",
                                  style="Panel.TLabelframe")
        plot_box.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        plot_box.columnconfigure(0, weight=1)
        plot_box.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 3), dpi=100, facecolor=self.PANEL)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._plot_placeholder()

        # ----- status bar -----
        bar = tk.Frame(self.root, bg=self.BG)
        bar.pack(fill="x", padx=16, pady=(4, 10))
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self.status_var, bg=self.BG, fg=self.MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self.progress.pack(side="right")

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
        n_frames = spec_db.shape[1]
        t_max = n_frames * HOP_LENGTH / SAMPLE_RATE
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

    # ---------- task runner (keeps UI responsive) ----------
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

    # ---------- library actions ----------
    def refresh_library(self):
        for item in self.lib.get_children():
            self.lib.delete(item)
        db = get_db()
        try:
            rows = db.execute(
                "SELECT id, name, num_hashes, added_at FROM songs ORDER BY name COLLATE NOCASE"
            ).fetchall()
        finally:
            db.close()
        for sid, name, n, added in rows:
            self.lib.insert("", "end", iid=str(sid), values=(name, f"{n:,}", added))
        total = sum(r[2] for r in rows)
        self.stats_var.set(f"{len(rows)} songs   |   {total:,} fingerprints stored")

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select songs",
            filetypes=[("Audio files", AUDIO_PATTERNS), ("All files", "*.*")],
        )
        if paths:
            self.start_adding(list(paths))

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select a folder of songs")
        if not folder:
            return
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(AUDIO_EXT)]
        if not paths:
            messagebox.showinfo("No songs", "No audio files were found in that folder.")
            return
        self.start_adding(paths)

    def start_adding(self, paths):
        def job():
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
            return added, skipped, failed

        self.run_task(job, self.after_add, "Adding songs...")

    def after_add(self, res):
        added, skipped, failed = res
        self.refresh_library()
        msg = f"Added {added}, skipped {skipped} (already in library)"
        if failed:
            msg += f", failed {failed}"
        self.set_status(msg)

    def remove_selected(self):
        if self.busy:
            return
        sel = self.lib.selection()
        if not sel:
            messagebox.showinfo("Nothing selected", "Select one or more songs to remove.")
            return
        if not messagebox.askyesno("Remove songs", f"Remove {len(sel)} song(s) from the library?"):
            return
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

    # ---------- identify actions ----------
    def identify_file(self):
        path = filedialog.askopenfilename(
            title="Choose an audio clip",
            filetypes=[("Audio files", AUDIO_PATTERNS), ("All files", "*.*")],
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
            self.detail_var.set("Add some songs first using 'Add Songs' or 'Add Folder'.")
            return

        if res["n_hashes"] == 0:
            self.result_var.set("Could not analyse the audio")
            self.result_lbl.configure(fg=self.BAD)
            self.detail_var.set("The clip may be silent or too short. Try again with a louder/longer clip.")
            return

        matches = res["matches"]
        for rank, (name, score, pct) in enumerate(matches, 1):
            self.res.insert("", "end", values=(rank, name, score, f"{pct:.1f}%"))

        if matches and matches[0][1] >= MIN_MATCHES and matches[0][2] >= MIN_MATCH_PCT:
            self.result_var.set(f"Match: {matches[0][0]}")
            self.result_lbl.configure(fg=self.GOOD)
            self.detail_var.set(
                f"{matches[0][1]} fingerprints lined up at the same time offset "
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
