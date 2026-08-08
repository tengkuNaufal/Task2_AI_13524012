"""
Algoritma Local Search untuk persoalan penjadwalan mata kuliah.

Seluruh algoritma bekerja pada ``TimetableProblem`` dari ``problem.py`` dan
memakai konvensi **minimasi**: nilai objective yang lebih kecil berarti jadwal
lebih baik. "Menanjak" (uphill) pada tulisan Hill-Climbing klasik berarti
*menurun* pada implementasi ini.

Algoritma yang tersedia
-----------------------
=====================================  ===========================================
Fungsi                                 Keterangan
=====================================  ===========================================
``hill_climbing_steepest``             Basic / Steepest-Ascent Hill-Climbing
``hill_climbing_sideways``             Hill-Climbing with Sideways Move
``hill_climbing_stochastic``           Stochastic Hill-Climbing
``hill_climbing_random_restart``       Random-Restart Hill-Climbing
``simulated_annealing``                Simulated Annealing (geometric cooling)
``genetic_algorithm``                  Genetic Algorithm
=====================================  ===========================================
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .problem import Cost, State, TimetableProblem

# --------------------------------------------------------------------------- #


@dataclass
class SearchResult:
    """Rangkuman satu kali eksekusi algoritma."""

    name: str
    initial_state: State
    final_state: State
    initial_cost: Cost
    final_cost: Cost
    history: List[int] = field(default_factory=list)
    iterations: int = 0
    evaluations: int = 0
    duration: float = 0.0
    extras: Dict[str, object] = field(default_factory=dict)

    @property
    def improvement(self) -> int:
        return self.initial_cost.total - self.final_cost.total

    def summary(self) -> str:
        if self.final_cost.feasible:
            status = "FEASIBLE"
        else:
            status = f"tidak feasible, {self.final_cost.hard_violations} pelanggaran keras"
        lines = [
            f"[{self.name}]",
            f"  objective awal   : {self.initial_cost.total}",
            f"  objective akhir  : {self.final_cost.total}  ({status})",
            f"  perbaikan        : {self.improvement}",
            f"  iterasi          : {self.iterations}",
            f"  evaluasi f(X)    : {self.evaluations}",
            f"  durasi           : {self.duration:.3f} s",
        ]
        for k, v in self.extras.items():
            if isinstance(v, (list, tuple)) and len(v) > 6:
                continue
            lines.append(f"  {k:<17}: {v}")
        return "\n".join(lines)


ProgressFn = Optional[Callable[[int, int], None]]


def _tick(progress: ProgressFn, it: int, value: int) -> None:
    if progress is not None:
        progress(it, value)


# --------------------------------------------------------------------------- #
# 1. Basic (Steepest-Ascent) Hill-Climbing
# --------------------------------------------------------------------------- #


def hill_climbing_steepest(
    problem: TimetableProblem,
    initial: State,
    max_iter: int = 5_000,
    progress: ProgressFn = None,
) -> SearchResult:
    """Steepest-Ascent HC — setiap iterasi memeriksa **seluruh** neighborhood.

    Pindah ke tetangga dengan objective terkecil; berhenti begitu tidak ada
    tetangga yang lebih baik (local optimum / plateau).
    """
    t0 = time.perf_counter()
    current = list(initial)
    current_val = problem.objective(current)
    init_cost = problem.evaluate(current)
    history = [current_val]
    evals = 1
    it = 0

    while it < max_iter:
        best_state: Optional[State] = None
        best_val = current_val
        best_move = None
        for nxt, move in problem.neighbors(current):
            val = problem.objective(nxt)
            evals += 1
            if val < best_val:
                best_val, best_state, best_move = val, nxt, move
        if best_state is None:  # tidak ada tetangga yang lebih baik
            break
        current, current_val = best_state, best_val
        it += 1
        history.append(current_val)
        _tick(progress, it, current_val)

    return SearchResult(
        name="Hill-Climbing (Steepest-Ascent)",
        initial_state=list(initial),
        final_state=current,
        initial_cost=init_cost,
        final_cost=problem.evaluate(current),
        history=history,
        iterations=it,
        evaluations=evals,
        duration=time.perf_counter() - t0,
        extras={"berhenti_karena": "local optimum" if it < max_iter else "batas iterasi"},
    )


# --------------------------------------------------------------------------- #
# 2. Hill-Climbing with Sideways Move
# --------------------------------------------------------------------------- #


def hill_climbing_sideways(
    problem: TimetableProblem,
    initial: State,
    max_sideways: int = 30,
    max_iter: int = 5_000,
    rng: Optional[random.Random] = None,
    progress: ProgressFn = None,
) -> SearchResult:
    """Steepest-Ascent HC yang boleh melangkah **mendatar** (objective sama).

    Berguna untuk menembus *plateau*. Agar tidak berputar selamanya, jumlah
    langkah mendatar berturut-turut dibatasi ``max_sideways``.
    """
    rng = rng or random.Random()
    t0 = time.perf_counter()
    current = list(initial)
    current_val = problem.objective(current)
    init_cost = problem.evaluate(current)
    history = [current_val]
    evals = 1
    it = 0
    sideways_used = 0
    sideways_streak = 0

    while it < max_iter:
        best_val = current_val
        best_states: List[State] = []
        equal_states: List[State] = []
        for nxt, _move in problem.neighbors(current):
            val = problem.objective(nxt)
            evals += 1
            if val < best_val:
                best_val = val
                best_states = [nxt]
                equal_states = []
            elif val == best_val and best_states:
                best_states.append(nxt)
            elif val == current_val and not best_states:
                equal_states.append(nxt)

        if best_states:  # ada perbaikan sesungguhnya
            current = rng.choice(best_states)
            current_val = best_val
            sideways_streak = 0
        elif equal_states and sideways_streak < max_sideways:
            current = rng.choice(equal_states)  # langkah mendatar
            sideways_streak += 1
            sideways_used += 1
        else:
            break

        it += 1
        history.append(current_val)
        _tick(progress, it, current_val)

    return SearchResult(
        name="Hill-Climbing (Sideways Move)",
        initial_state=list(initial),
        final_state=current,
        initial_cost=init_cost,
        final_cost=problem.evaluate(current),
        history=history,
        iterations=it,
        evaluations=evals,
        duration=time.perf_counter() - t0,
        extras={
            "sideways_terpakai": sideways_used,
            "batas_sideways": max_sideways,
        },
    )


# --------------------------------------------------------------------------- #
# 3. Stochastic Hill-Climbing
# --------------------------------------------------------------------------- #


def hill_climbing_stochastic(
    problem: TimetableProblem,
    initial: State,
    max_iter: int = 20_000,
    rng: Optional[random.Random] = None,
    progress: ProgressFn = None,
) -> SearchResult:
    """Stochastic HC — tiap iterasi mengundi **satu** tetangga acak.

    Tetangga diterima hanya bila lebih baik. Karena tidak pernah memeriksa
    seluruh neighborhood, satu iterasinya jauh lebih murah daripada
    Steepest-Ascent, sehingga jumlah iterasi ``max_iter`` dijadikan kriteria
    berhenti.
    """
    rng = rng or random.Random()
    t0 = time.perf_counter()
    current = list(initial)
    current_val = problem.objective(current)
    init_cost = problem.evaluate(current)
    history = [current_val]
    evals = 1
    accepted = 0

    for it in range(1, max_iter + 1):
        nxt, _move = problem.random_neighbor(current, rng)
        val = problem.objective(nxt)
        evals += 1
        if val < current_val:
            current, current_val = nxt, val
            accepted += 1
        history.append(current_val)
        _tick(progress, it, current_val)

    return SearchResult(
        name="Hill-Climbing (Stochastic)",
        initial_state=list(initial),
        final_state=current,
        initial_cost=init_cost,
        final_cost=problem.evaluate(current),
        history=history,
        iterations=max_iter,
        evaluations=evals,
        duration=time.perf_counter() - t0,
        extras={
            "move_diterima": accepted,
            "rasio_penerimaan": round(accepted / max(1, max_iter), 4),
        },
    )


# --------------------------------------------------------------------------- #
# 4. Random-Restart Hill-Climbing
# --------------------------------------------------------------------------- #


def hill_climbing_random_restart(
    problem: TimetableProblem,
    restarts: int = 5,
    max_iter: int = 5_000,
    rng: Optional[random.Random] = None,
    initial: Optional[State] = None,
    progress: ProgressFn = None,
) -> SearchResult:
    """Menjalankan Steepest-Ascent HC berulang kali dari state awal acak.

    ``initial`` (bila diberikan) dipakai sebagai state awal restart pertama
    agar hasilnya sebanding dengan varian HC lain.
    """
    rng = rng or random.Random()
    t0 = time.perf_counter()

    best: Optional[SearchResult] = None
    per_restart: List[int] = []
    history: List[int] = []
    total_evals = 0
    total_iters = 0

    for k in range(restarts):
        start = list(initial) if (k == 0 and initial is not None) else problem.random_state(rng)
        run = hill_climbing_steepest(problem, start, max_iter=max_iter)
        per_restart.append(run.final_cost.total)
        history.extend(run.history)
        total_evals += run.evaluations
        total_iters += run.iterations
        if best is None or run.final_cost.total < best.final_cost.total:
            best = run
        _tick(progress, k + 1, run.final_cost.total)

    assert best is not None
    first_initial = list(initial) if initial is not None else best.initial_state
    return SearchResult(
        name=f"Hill-Climbing (Random Restart, {restarts}x)",
        initial_state=first_initial,
        final_state=best.final_state,
        initial_cost=problem.evaluate(first_initial),
        final_cost=best.final_cost,
        history=history,
        iterations=total_iters,
        evaluations=total_evals,
        duration=time.perf_counter() - t0,
        extras={
            "jumlah_restart": restarts,
            "objective_per_restart": per_restart,
            "restart_terbaik": per_restart.index(min(per_restart)) + 1,
            "iterasi_per_restart": total_iters / restarts,
        },
    )


# --------------------------------------------------------------------------- #
# 5. Simulated Annealing
# --------------------------------------------------------------------------- #


def simulated_annealing(
    problem: TimetableProblem,
    initial: State,
    t0_temp: float = 60.0,
    alpha: float = 0.9995,
    t_min: float = 1e-3,
    max_iter: int = 60_000,
    rng: Optional[random.Random] = None,
    progress: ProgressFn = None,
) -> SearchResult:
    """Simulated Annealing dengan *geometric cooling* ``T <- alpha * T``.

    Tetangga yang lebih buruk tetap dapat diterima dengan peluang
    ``exp(-dE / T)`` (dE = kenaikan objective). Peluang ini besar saat suhu
    masih tinggi (eksplorasi) dan mengecil seiring pendinginan (eksploitasi),
    sehingga SA mampu keluar dari local optimum yang menjebak Hill-Climbing.
    """
    rng = rng or random.Random()
    start = time.perf_counter()
    current = list(initial)
    current_val = problem.objective(current)
    init_cost = problem.evaluate(current)
    best_state, best_val = list(current), current_val

    history = [current_val]
    best_history = [best_val]
    temps: List[float] = []
    accept_probs: List[Tuple[int, float]] = []  # (iterasi, exp(-dE/T)) untuk move memburuk

    temp = t0_temp
    evals = 1
    accepted_worse = 0
    rejected_worse = 0
    stuck_events = 0  # berapa kali SA "terjebak": menolak move memburuk saat sudah dingin
    it = 0

    while it < max_iter and temp > t_min:
        it += 1
        nxt, _move = problem.random_neighbor(current, rng)
        val = problem.objective(nxt)
        evals += 1
        delta = val - current_val

        if delta <= 0:
            current, current_val = nxt, val
        else:
            p = math.exp(-delta / temp)
            accept_probs.append((it, p))
            if rng.random() < p:
                current, current_val = nxt, val
                accepted_worse += 1
            else:
                rejected_worse += 1
                if temp < 1.0:
                    stuck_events += 1

        if current_val < best_val:
            best_state, best_val = list(current), current_val

        history.append(current_val)
        best_history.append(best_val)
        temps.append(temp)
        temp *= alpha
        _tick(progress, it, current_val)

    return SearchResult(
        name="Simulated Annealing",
        initial_state=list(initial),
        final_state=best_state,
        initial_cost=init_cost,
        final_cost=problem.evaluate(best_state),
        history=history,
        iterations=it,
        evaluations=evals,
        duration=time.perf_counter() - start,
        extras={
            "T0": t0_temp,
            "alpha": alpha,
            "T_akhir": round(temp, 6),
            "move_memburuk_diterima": accepted_worse,
            "move_memburuk_ditolak": rejected_worse,
            "stuck_di_suhu_rendah": stuck_events,
            "best_history": best_history,
            "temperatures": temps,
            "accept_probs": accept_probs,
        },
    )


# --------------------------------------------------------------------------- #
# 6. Genetic Algorithm
# --------------------------------------------------------------------------- #


def _tournament(
    pop: Sequence[State], vals: Sequence[int], k: int, rng: random.Random
) -> State:
    """Tournament selection: ambil ``k`` individu acak, kembalikan yang terbaik."""
    best_idx = rng.randrange(len(pop))
    for _ in range(k - 1):
        idx = rng.randrange(len(pop))
        if vals[idx] < vals[best_idx]:
            best_idx = idx
    return list(pop[best_idx])


def _uniform_crossover(
    a: State, b: State, rng: random.Random
) -> Tuple[State, State]:
    """Uniform crossover pada level *gen* (satu gen = satu pasangan (slot, ruang))."""
    c1, c2 = list(a), list(b)
    for i in range(len(a)):
        if rng.random() < 0.5:
            c1[i], c2[i] = b[i], a[i]
    return c1, c2


def _one_point_crossover(
    a: State, b: State, rng: random.Random
) -> Tuple[State, State]:
    p = rng.randrange(1, len(a)) if len(a) > 1 else 0
    return list(a[:p]) + list(b[p:]), list(b[:p]) + list(a[p:])


def genetic_algorithm(
    problem: TimetableProblem,
    population_size: int = 80,
    generations: int = 300,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.15,
    elitism: int = 2,
    tournament_k: int = 3,
    crossover: str = "uniform",
    rng: Optional[random.Random] = None,
    initial: Optional[State] = None,
    progress: ProgressFn = None,
) -> SearchResult:
    """Genetic Algorithm untuk penjadwalan.

    * **Kromosom** — persis representasi state: vektor ``n`` gen, gen ke-``i``
      adalah pasangan ``(slot, ruang)`` mata kuliah ke-``i``. Panjang kromosom
      tetap, sehingga crossover dan mutasi selalu menghasilkan jadwal utuh yang
      valid secara struktur.
    * **Fitness** — ``-f(X)``; makin besar makin baik (f(X) diminimasi).
    * **Selection** — tournament selection berukuran ``tournament_k``.
    * **Crossover** — ``uniform`` (default) atau ``one_point``.
    * **Mutation** — dengan peluang ``mutation_rate`` sebuah gen acak diganti
      pasangan ``(slot, ruang)`` acak.
    * **Elitism** — ``elitism`` individu terbaik disalin apa adanya ke generasi
      berikutnya agar solusi terbaik tidak pernah hilang.
    """
    rng = rng or random.Random()
    t0 = time.perf_counter()
    cx = _uniform_crossover if crossover == "uniform" else _one_point_crossover

    population: List[State] = []
    if initial is not None:
        population.append(list(initial))
    while len(population) < population_size:
        population.append(problem.random_state(rng))

    vals = [problem.objective(ind) for ind in population]
    evals = len(vals)

    init_state = list(initial) if initial is not None else list(population[0])
    init_cost = problem.evaluate(init_state)

    best_idx = min(range(len(vals)), key=vals.__getitem__)
    best_state, best_val = list(population[best_idx]), vals[best_idx]

    best_history = [best_val]
    avg_history = [sum(vals) / len(vals)]
    worst_history = [max(vals)]

    for gen in range(1, generations + 1):
        order = sorted(range(len(vals)), key=vals.__getitem__)
        new_pop: List[State] = [list(population[i]) for i in order[:elitism]]

        while len(new_pop) < population_size:
            p1 = _tournament(population, vals, tournament_k, rng)
            p2 = _tournament(population, vals, tournament_k, rng)
            if rng.random() < crossover_rate:
                c1, c2 = cx(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            for child in (c1, c2):
                if rng.random() < mutation_rate:
                    i = rng.randrange(problem.n)
                    child[i] = (
                        rng.randrange(problem.n_slots),
                        rng.randrange(problem.n_rooms),
                    )
                if len(new_pop) < population_size:
                    new_pop.append(child)

        population = new_pop
        vals = [problem.objective(ind) for ind in population]
        evals += len(vals)

        gen_best_idx = min(range(len(vals)), key=vals.__getitem__)
        if vals[gen_best_idx] < best_val:
            best_val = vals[gen_best_idx]
            best_state = list(population[gen_best_idx])

        best_history.append(best_val)
        avg_history.append(sum(vals) / len(vals))
        worst_history.append(max(vals))
        _tick(progress, gen, best_val)

    return SearchResult(
        name="Genetic Algorithm",
        initial_state=init_state,
        final_state=best_state,
        initial_cost=init_cost,
        final_cost=problem.evaluate(best_state),
        history=best_history,
        iterations=generations,
        evaluations=evals,
        duration=time.perf_counter() - t0,
        extras={
            "ukuran_populasi": population_size,
            "generasi": generations,
            "crossover": crossover,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elitism": elitism,
            "tournament_k": tournament_k,
            "avg_history": avg_history,
            "worst_history": worst_history,
        },
    )


# --------------------------------------------------------------------------- #

ALGORITHMS = {
    "hc-steepest": "Basic / Steepest-Ascent Hill-Climbing",
    "hc-sideways": "Hill-Climbing with Sideways Move",
    "hc-stochastic": "Stochastic Hill-Climbing",
    "hc-random-restart": "Random-Restart Hill-Climbing",
    "sa": "Simulated Annealing",
    "ga": "Genetic Algorithm",
}
