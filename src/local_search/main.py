"""
Proof of Concept — Local Search untuk Penjadwalan Mata Kuliah.

Cara pakai (dijalankan dari direktori ``src/``)::

    python -m local_search.main --algo hc-steepest --seed 42
    python -m local_search.main --algo sa --seed 42 --sa-iter 60000
    python -m local_search.main --algo ga --seed 42 --pop 80 --gen 300
    python -m local_search.main --algo all --seed 42 --plot --animate

Perintah utama tersedia lewat ``--algo``:

======================  ===================================================
Nilai                   Algoritma
======================  ===================================================
``hc-steepest``         Basic / Steepest-Ascent Hill-Climbing
``hc-sideways``         Hill-Climbing with Sideways Move
``hc-stochastic``       Stochastic Hill-Climbing
``hc-random-restart``   Random-Restart Hill-Climbing
``sa``                  Simulated Annealing
``ga``                  Genetic Algorithm
``all``                 Menjalankan keenamnya dari state awal yang sama
======================  ===================================================
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):  # memungkinkan `python src/local_search/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_search import algorithms as algo
from local_search import visualize as viz
from local_search.problem import TimetableProblem, default_problem

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results" / "local_search"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="local_search",
        description="PoC Local Search — Penjadwalan Mata Kuliah",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--algo",
        default="all",
        choices=list(algo.ALGORITHMS) + ["all"],
        help="algoritma yang dijalankan",
    )
    p.add_argument("--seed", type=int, default=42, help="seed RNG (reprodusibilitas)")
    p.add_argument("--instance", type=Path, default=None, help="berkas JSON instansi")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="direktori keluaran")

    p.add_argument("--max-iter", type=int, default=5000, help="batas iterasi HC")
    p.add_argument("--max-sideways", type=int, default=40, help="batas langkah sideways")
    p.add_argument("--stochastic-iter", type=int, default=30000, help="iterasi Stochastic HC")
    p.add_argument("--restarts", type=int, default=5, help="jumlah restart untuk Random Restart")

    p.add_argument("--sa-t0", type=float, default=60.0, help="suhu awal SA")
    p.add_argument("--sa-alpha", type=float, default=0.9997, help="laju pendinginan SA")
    p.add_argument("--sa-tmin", type=float, default=1e-3, help="suhu minimum SA")
    p.add_argument("--sa-iter", type=int, default=60000, help="batas iterasi SA")

    p.add_argument("--pop", type=int, default=80, help="ukuran populasi GA")
    p.add_argument("--gen", type=int, default=400, help="jumlah generasi GA")
    p.add_argument("--mutation", type=float, default=0.2, help="peluang mutasi GA")
    p.add_argument("--crossover-rate", type=float, default=0.9, help="peluang crossover GA")
    p.add_argument("--elitism", type=int, default=2, help="jumlah elite GA")
    p.add_argument(
        "--crossover", default="uniform", choices=["uniform", "one_point"],
        help="operator crossover GA",
    )

    p.add_argument("--plot", action="store_true", help="simpan grafik ke direktori keluaran")
    p.add_argument("--animate", action="store_true", help="simpan animasi GIF proses pencarian")
    p.add_argument("--frames", type=int, default=60, help="jumlah frame animasi")
    p.add_argument("--quiet", action="store_true", help="tekan cetak jadwal lengkap")
    p.add_argument(
        "--save-instance", type=Path, default=None,
        help="simpan instansi bawaan ke berkas JSON lalu keluar",
    )
    return p


def run_one(name: str, problem: TimetableProblem, initial, args, rng: random.Random):
    """Menjalankan satu algoritma sesuai kode ``name``."""
    if name == "hc-steepest":
        return algo.hill_climbing_steepest(problem, initial, max_iter=args.max_iter)
    if name == "hc-sideways":
        return algo.hill_climbing_sideways(
            problem, initial, max_sideways=args.max_sideways,
            max_iter=args.max_iter, rng=rng,
        )
    if name == "hc-stochastic":
        return algo.hill_climbing_stochastic(
            problem, initial, max_iter=args.stochastic_iter, rng=rng
        )
    if name == "hc-random-restart":
        return algo.hill_climbing_random_restart(
            problem, restarts=args.restarts, max_iter=args.max_iter,
            rng=rng, initial=initial,
        )
    if name == "sa":
        return algo.simulated_annealing(
            problem, initial, t0_temp=args.sa_t0, alpha=args.sa_alpha,
            t_min=args.sa_tmin, max_iter=args.sa_iter, rng=rng,
        )
    if name == "ga":
        return algo.genetic_algorithm(
            problem, population_size=args.pop, generations=args.gen,
            crossover_rate=args.crossover_rate, mutation_rate=args.mutation,
            elitism=args.elitism, crossover=args.crossover, rng=rng, initial=initial,
        )
    raise ValueError(name)  # pragma: no cover


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.save_instance:
        default_problem().save(args.save_instance)
        print(f"Instansi bawaan disimpan ke {args.save_instance}")
        return 0

    problem = (
        TimetableProblem.load(args.instance) if args.instance else default_problem()
    )
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PoC LOCAL SEARCH — PENJADWALAN MATA KULIAH")
    print("=" * 78)
    print(f"Instansi          : {problem.name}")
    print(f"Mata kuliah (n)   : {problem.n}")
    print(f"Ruangan (|R|)     : {problem.n_rooms}")
    print(f"Slot waktu (|T|)  : {problem.n_slots}  (5 hari x 5 jam)")
    print(f"Ukuran ruang state: (|T| x |R|)^n = ({problem.n_slots} x {problem.n_rooms})^{problem.n}")
    print(f"Ukuran neighborhood: {problem.neighborhood_size()} state per iterasi")
    print(f"Seed              : {args.seed}")
    print()

    rng = random.Random(args.seed)
    initial = problem.random_state(rng)
    init_cost = problem.evaluate(initial)

    print("-" * 78)
    print("STATE AWAL (dibangkitkan secara random)")
    print("-" * 78)
    if not args.quiet:
        print(viz.render_timetable(problem, initial))
    print(init_cost.report())
    print()

    names = list(algo.ALGORITHMS) if args.algo == "all" else [args.algo]
    results = []
    for name in names:
        sub_rng = random.Random(args.seed + 1000 * (names.index(name) + 1))
        print("-" * 78)
        print(f"MENJALANKAN: {algo.ALGORITHMS[name]}")
        print("-" * 78)
        res = run_one(name, problem, initial, args, sub_rng)
        results.append(res)
        print(res.summary())
        print()

    best = min(results, key=lambda r: r.final_cost.total)

    print("=" * 78)
    print("PERBANDINGAN")
    print("=" * 78)
    print(
        f"{'Algoritma':<38} {'f(awal)':>8} {'f(akhir)':>9} "
        f"{'iterasi':>8} {'eval':>9} {'detik':>7} {'feasible':>9}"
    )
    print("-" * 92)
    for r in results:
        print(
            f"{r.name:<38} {r.initial_cost.total:>8} {r.final_cost.total:>9} "
            f"{r.iterations:>8} {r.evaluations:>9} {r.duration:>7.2f} "
            f"{('ya' if r.final_cost.feasible else 'tidak'):>9}"
        )
    print()

    print("=" * 78)
    print(f"STATE AKHIR TERBAIK — {best.name}")
    print("=" * 78)
    if not args.quiet:
        print(viz.render_timetable(problem, best.final_state))
        print(viz.render_assignment(problem, best.final_state))
        print()
    print(best.final_cost.report())
    print()
    print(viz.render_violations(problem, best.final_state))
    print()

    # --- keluaran berkas -------------------------------------------------- #
    summary = {
        "instance": problem.name,
        "seed": args.seed,
        "initial_objective": init_cost.total,
        "initial_counts": init_cost.counts,
        "results": [
            {
                "name": r.name,
                "final_objective": r.final_cost.total,
                "feasible": r.final_cost.feasible,
                "hard_violations": r.final_cost.hard_violations,
                "soft_penalty": r.final_cost.soft_penalty,
                "counts": r.final_cost.counts,
                "iterations": r.iterations,
                "evaluations": r.evaluations,
                "duration_s": round(r.duration, 3),
                "extras": {
                    k: v
                    for k, v in r.extras.items()
                    if k not in ("best_history", "temperatures", "accept_probs",
                                 "avg_history", "worst_history")
                },
            }
            for r in results
        ],
        "best": best.name,
        "best_state": best.final_state,
    }
    (out / f"summary_{args.algo}_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out / f"jadwal_terbaik_seed{args.seed}.txt").write_text(
        viz.render_timetable(problem, best.final_state)
        + "\n\n"
        + viz.render_assignment(problem, best.final_state)
        + "\n\n"
        + best.final_cost.report(),
        encoding="utf-8",
    )
    print(f"Ringkasan JSON  -> {out / f'summary_{args.algo}_seed{args.seed}.json'}")
    print(f"Jadwal terbaik  -> {out / f'jadwal_terbaik_seed{args.seed}.txt'}")

    if args.plot:
        p = viz.plot_history(results, out / f"objective_{args.algo}_seed{args.seed}.png")
        print(f"Grafik objective-> {p}")
        for r in results:
            if r.name.startswith("Simulated"):
                print(f"Diagnostik SA   -> {viz.plot_sa_diagnostics(r, out / 'sa_diagnostics.png')}")
            if r.name.startswith("Genetic"):
                print(f"Diagnostik GA   -> {viz.plot_ga_diagnostics(r, out / 'ga_diagnostics.png')}")
        print(f"Heatmap awal    -> {viz.plot_timetable_heatmap(problem, initial, out / 'heatmap_awal.png', 'State awal (random)')}")
        print(f"Heatmap akhir   -> {viz.plot_timetable_heatmap(problem, best.final_state, out / 'heatmap_akhir.png', f'State akhir — {best.name}')}")

    if args.animate:
        snaps = _collect_snapshots(problem, initial, args, rng)
        gif = viz.animate_search(problem, snaps, out / "animasi_pencarian.gif")
        print(f"Animasi         -> {gif if gif else '(gagal: Pillow tidak tersedia)'}")

    return 0


def _collect_snapshots(problem, initial, args, rng):
    """Jalankan ulang Simulated Annealing sambil merekam state tiap beberapa iterasi."""
    snaps = []
    state = list(initial)
    val = problem.objective(state)
    snaps.append((0, list(state), val))
    r = random.Random(args.seed + 7)
    temp = args.sa_t0
    import math

    every = max(1, args.sa_iter // args.frames)
    for it in range(1, args.sa_iter + 1):
        nxt, _ = problem.random_neighbor(state, r)
        v = problem.objective(nxt)
        d = v - val
        if d <= 0 or r.random() < math.exp(-d / max(temp, 1e-12)):
            state, val = nxt, v
        temp *= args.sa_alpha
        if it % every == 0:
            snaps.append((it, list(state), val))
    return snaps


if __name__ == "__main__":
    raise SystemExit(main())
