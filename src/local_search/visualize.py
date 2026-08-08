"""
Visualisasi hasil Local Search: render jadwal ke teks, plot objective vs
iterasi, plot khusus Simulated Annealing dan Genetic Algorithm, serta animasi
perubahan state antar iterasi.

Modul plot mengimpor matplotlib secara *lazy* sehingga proof of concept tetap
dapat dijalankan pada lingkungan tanpa matplotlib (keluaran teks saja).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .problem import (
    DAYS,
    N_PERIODS,
    PERIODS,
    State,
    TimetableProblem,
    slot_day,
    slot_period,
)

# --------------------------------------------------------------------------- #
# Render teks
# --------------------------------------------------------------------------- #


def render_timetable(problem: TimetableProblem, state: State, width: int = 22) -> str:
    """Cetak jadwal sebagai matriks hari x jam; sel berisi ``KODE@RUANG``."""
    grid: Dict[Tuple[int, int], List[str]] = {}
    for i, (t, r) in enumerate(state):
        key = (slot_day(t), slot_period(t))
        grid.setdefault(key, []).append(
            f"{problem.courses[i].code}@{problem.rooms[r].name}"
        )

    header = f"{'Jam':<13}" + "".join(f"{d:<{width}}" for d in DAYS)
    lines = [header, "-" * len(header)]
    for p in range(N_PERIODS):
        max_rows = max((len(grid.get((d, p), [])) for d in range(len(DAYS))), default=0)
        max_rows = max(max_rows, 1)
        for row in range(max_rows):
            label = PERIODS[p] if row == 0 else ""
            line = f"{label:<13}"
            for d in range(len(DAYS)):
                cell = grid.get((d, p), [])
                line += f"{(cell[row] if row < len(cell) else '-'):<{width}}"
            lines.append(line.rstrip())
        lines.append("")
    return "\n".join(lines)


def render_assignment(problem: TimetableProblem, state: State) -> str:
    """Daftar penugasan per mata kuliah, diurutkan menurut waktu."""
    rows = sorted(range(problem.n), key=lambda i: (state[i][0], state[i][1]))
    lines = [
        f"{'Kode':<10} {'Mata Kuliah':<38} {'Dosen':<22} "
        f"{'Hari':<8} {'Jam':<12} {'Ruang':<12} {'Mhs':>4} {'Kap':>4}"
    ]
    lines.append("-" * len(lines[0]))
    for i in rows:
        t, r = state[i]
        c = problem.courses[i]
        lines.append(
            f"{c.code:<10} {c.name[:38]:<38} {problem.lecturers[c.lecturer][:22]:<22} "
            f"{DAYS[slot_day(t)]:<8} {PERIODS[slot_period(t)]:<12} "
            f"{problem.rooms[r].name:<12} {c.students:>4} {problem.rooms[r].capacity:>4}"
        )
    return "\n".join(lines)


def render_violations(problem: TimetableProblem, state: State, limit: int = 12) -> str:
    """Rincian pelanggaran hard constraint yang tersisa (untuk audit hasil)."""
    msgs: List[str] = []
    seen_room: Dict[Tuple[int, int], List[int]] = {}
    seen_lect: Dict[Tuple[int, int], List[int]] = {}
    seen_group: Dict[Tuple[int, int], List[int]] = {}

    for i, (t, r) in enumerate(state):
        seen_room.setdefault((t, r), []).append(i)
        seen_lect.setdefault((t, problem.courses[i].lecturer), []).append(i)
        for g in problem.courses[i].groups:
            seen_group.setdefault((t, g), []).append(i)

    for (t, r), ids in seen_room.items():
        if len(ids) > 1:
            codes = ", ".join(problem.courses[i].code for i in ids)
            msgs.append(f"[H1] Ruang {problem.rooms[r].name} bentrok pada slot {t}: {codes}")
    for (t, l), ids in seen_lect.items():
        if len(ids) > 1:
            codes = ", ".join(problem.courses[i].code for i in ids)
            msgs.append(f"[H2] {problem.lecturers[l]} bentrok pada slot {t}: {codes}")
    for (t, g), ids in seen_group.items():
        if len(ids) > 1:
            codes = ", ".join(problem.courses[i].code for i in ids)
            msgs.append(f"[H3] Kelompok {problem.groups[g]} bentrok pada slot {t}: {codes}")
    for i, (t, r) in enumerate(state):
        c, room = problem.courses[i], problem.rooms[r]
        if c.students > room.capacity:
            msgs.append(
                f"[H4] {c.code} ({c.students} mhs) melebihi kapasitas "
                f"{room.name} ({room.capacity})"
            )
        if c.needs_lab and room.kind != "LAB":
            msgs.append(f"[H5] {c.code} butuh lab tetapi ditempatkan di {room.name}")

    if not msgs:
        return "Tidak ada pelanggaran hard constraint. Jadwal FEASIBLE."
    out = msgs[:limit]
    if len(msgs) > limit:
        out.append(f"... dan {len(msgs) - limit} pelanggaran lain")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_history(
    results: Sequence["object"],
    path: str | Path,
    title: str = "Nilai objective function terhadap iterasi",
    logy: bool = True,
) -> Path:
    """Plot f(X) vs iterasi untuk satu atau beberapa ``SearchResult``.

    Panjang lintasan tiap algoritma berbeda ekstrem (HC berhenti di puluhan
    iterasi, SA berjalan puluhan ribu), sehingga dibuat dua panel: panel kiri
    memakai sumbu-x logaritmik untuk melihat keseluruhan, panel kanan
    memperbesar 300 iterasi pertama agar lintasan varian Hill-Climbing terbaca.
    """
    plt = _plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for res in results:
        label = f"{res.name} (akhir={res.final_cost.total})"
        ax.plot(range(1, len(res.history) + 1), res.history, label=label, lw=1.3)
        ax2.plot(res.history[:300], lw=1.3)

    ax.set_xscale("log")
    ax.set_xlabel("Iterasi (skala log)")
    ax.set_ylabel("f(X)  (semakin kecil semakin baik)")
    ax.set_title("Seluruh lintasan")
    ax2.set_xlabel("Iterasi")
    ax2.set_title("Perbesaran 300 iterasi pertama")
    for a in (ax, ax2):
        if logy:
            # f(X) selalu >= 0; linthresh menahan sumbu agar tidak melebar ke negatif
            a.set_yscale("symlog", linthresh=10)
            a.set_ylim(bottom=0)
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_sa_diagnostics(result, path: str | Path) -> Path:
    """Tiga panel khas Simulated Annealing: f(X), suhu, dan peluang penerimaan."""
    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(result.history, lw=0.7, label="f(X) saat ini")
    axes[0].plot(result.extras["best_history"], lw=1.4, label="f(X) terbaik")
    axes[0].set_ylabel("f(X)")
    axes[0].set_yscale("symlog")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Simulated Annealing — objective, suhu, dan peluang penerimaan")

    axes[1].plot(result.extras["temperatures"], color="tab:red", lw=1.2)
    axes[1].set_ylabel("Suhu T")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    probs = result.extras["accept_probs"]
    if probs:
        xs = [p[0] for p in probs]
        ys = [p[1] for p in probs]
        axes[2].scatter(xs, ys, s=2, alpha=0.25, color="tab:green")
    axes[2].set_ylabel(r"$e^{-\Delta E / T}$")
    axes[2].set_xlabel("Iterasi")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_ga_diagnostics(result, path: str | Path) -> Path:
    """Kurva f(X) terbaik, rata-rata, dan terburuk populasi tiap generasi."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(result.history, label="terbaik", lw=1.6)
    ax.plot(result.extras["avg_history"], label="rata-rata populasi", lw=1.2)
    ax.plot(result.extras["worst_history"], label="terburuk", lw=0.9, alpha=0.7)
    ax.set_xlabel("Generasi")
    ax.set_ylabel("f(X)")
    ax.set_yscale("symlog")
    ax.set_title("Genetic Algorithm — perkembangan populasi tiap generasi")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_timetable_heatmap(
    problem: TimetableProblem, state: State, path: str | Path, title: str = ""
) -> Path:
    """Peta okupansi ruangan (baris) terhadap slot waktu (kolom)."""
    plt = _plt()
    occ = [[0] * problem.n_slots for _ in range(problem.n_rooms)]
    for t, r in state:
        occ[r][t] += 1

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(occ, aspect="auto", cmap="YlGnBu", vmin=0, vmax=2)
    ax.set_yticks(range(problem.n_rooms))
    ax.set_yticklabels([r.name for r in problem.rooms], fontsize=8)
    ax.set_xticks(range(0, problem.n_slots, N_PERIODS))
    ax.set_xticklabels(DAYS, fontsize=9)
    for d in range(1, len(DAYS)):
        ax.axvline(d * N_PERIODS - 0.5, color="white", lw=1.5)
    ax.set_title(title or "Okupansi ruangan per slot waktu")
    fig.colorbar(im, ax=ax, label="jumlah kelas (>1 = bentrok)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def animate_search(
    problem: TimetableProblem,
    snapshots: Sequence[Tuple[int, State, int]],
    path: str | Path,
    fps: int = 6,
) -> Optional[Path]:
    """Animasi GIF perubahan state antar iterasi.

    ``snapshots`` berisi tuple ``(iterasi, state, nilai_objective)``.
    Mengembalikan ``None`` bila writer GIF tidak tersedia.
    """
    plt = _plt()
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    occ0 = [[0] * problem.n_slots for _ in range(problem.n_rooms)]
    im = ax.imshow(occ0, aspect="auto", cmap="YlGnBu", vmin=0, vmax=2)
    ax.set_yticks(range(problem.n_rooms))
    ax.set_yticklabels([r.name for r in problem.rooms], fontsize=7)
    ax.set_xticks(range(0, problem.n_slots, N_PERIODS))
    ax.set_xticklabels(DAYS, fontsize=8)
    for d in range(1, len(DAYS)):
        ax.axvline(d * N_PERIODS - 0.5, color="white", lw=1.2)

    xs = [s[0] for s in snapshots]
    ys = [s[2] for s in snapshots]
    ax2.plot(xs, ys, lw=1.0, color="tab:gray")
    (marker,) = ax2.plot([], [], "o", color="tab:red")
    ax2.set_yscale("symlog")
    ax2.set_xlabel("Iterasi")
    ax2.set_ylabel("f(X)")
    ax2.grid(alpha=0.3)

    def update(k: int):
        it, state, val = snapshots[k]
        occ = [[0] * problem.n_slots for _ in range(problem.n_rooms)]
        for t, r in state:
            occ[r][t] += 1
        im.set_data(occ)
        ax.set_title(f"Iterasi {it} — f(X) = {val}")
        marker.set_data([it], [val])
        return im, marker

    anim = FuncAnimation(fig, update, frames=len(snapshots), blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(path, writer=PillowWriter(fps=fps))
    except Exception:  # pragma: no cover - Pillow tidak tersedia
        plt.close(fig)
        return None
    plt.close(fig)
    return path
