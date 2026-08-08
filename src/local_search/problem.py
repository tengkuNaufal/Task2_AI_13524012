"""
Penjadwalan Mata Kuliah (University Course Timetabling) sebagai persoalan Local Search.

Modul ini memuat seluruh *definisi masalah*:

* representasi ``state`` (complete-state formulation),
* pembangkitan ``initial state`` secara random,
* ``objective function`` (fungsi biaya) beserta dekomposisinya per komponen,
* ``successor function`` beserta tiga jenis ``move`` yang diperbolehkan.

Algoritma pencariannya sendiri berada di ``algorithms.py`` sehingga definisi
masalah dan strategi pencarian benar-benar terpisah.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Dimensi waktu
# --------------------------------------------------------------------------- #

DAYS: Tuple[str, ...] = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat")
PERIODS: Tuple[str, ...] = (
    "07.00-09.00",
    "09.00-11.00",
    "11.00-13.00",
    "13.00-15.00",
    "15.00-17.00",
)
N_DAYS = len(DAYS)
N_PERIODS = len(PERIODS)
N_SLOTS = N_DAYS * N_PERIODS  # 25 slot waktu


def slot_day(t: int) -> int:
    """Indeks hari dari sebuah slot waktu."""
    return t // N_PERIODS


def slot_period(t: int) -> int:
    """Indeks jam ke-berapa dalam hari dari sebuah slot waktu."""
    return t % N_PERIODS


def slot_name(t: int) -> str:
    return f"{DAYS[slot_day(t)]} {PERIODS[slot_period(t)]}"


# --------------------------------------------------------------------------- #
# Entitas masalah
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Room:
    """Ruangan. ``kind`` bernilai ``"KULIAH"`` atau ``"LAB"``."""

    id: int
    name: str
    capacity: int
    kind: str


@dataclass(frozen=True)
class Course:
    """Satu kelas mata kuliah yang harus dijadwalkan tepat satu kali."""

    id: int
    code: str
    name: str
    lecturer: int          # indeks dosen pengampu
    groups: Tuple[int, ...]  # kelompok mahasiswa (prodi-angkatan-kelas) peserta
    students: int
    needs_lab: bool


@dataclass
class Weights:
    """Bobot tiap komponen fungsi objektif.

    Bobot *hard constraint* dibuat satu orde lebih besar daripada *soft
    constraint* sehingga satu pelanggaran keras tidak pernah dapat "dibayar"
    oleh sejumlah perbaikan lunak.
    """

    # hard constraints
    room_clash: int = 100
    lecturer_clash: int = 100
    group_clash: int = 100
    capacity: int = 60
    room_type: int = 80
    # soft constraints
    lecturer_unavailable: int = 10
    late_slot: int = 3
    lecturer_overload: int = 5
    group_overload: int = 4
    group_gap: int = 2
    lecturer_gap: int = 2
    room_waste: int = 1
    lab_misuse: int = 4

    HARD_KEYS = ("room_clash", "lecturer_clash", "group_clash", "capacity", "room_type")
    SOFT_KEYS = (
        "lecturer_unavailable",
        "late_slot",
        "lecturer_overload",
        "group_overload",
        "group_gap",
        "lecturer_gap",
        "room_waste",
        "lab_misuse",
    )


@dataclass
class Cost:
    """Hasil evaluasi sebuah state, lengkap dengan rincian per komponen."""

    total: int
    counts: Dict[str, int]
    weights: Weights

    @property
    def hard_violations(self) -> int:
        return sum(self.counts[k] for k in Weights.HARD_KEYS)

    @property
    def soft_penalty(self) -> int:
        return sum(self.counts[k] * getattr(self.weights, k) for k in Weights.SOFT_KEYS)

    @property
    def feasible(self) -> bool:
        return self.hard_violations == 0

    def report(self) -> str:
        lines = [f"Objective f(X) = {self.total}"]
        lines.append("  Hard constraints (bobot besar):")
        for k in Weights.HARD_KEYS:
            w = getattr(self.weights, k)
            lines.append(
                f"    {k:<22} {self.counts[k]:>4} x {w:<4} = {self.counts[k] * w:>6}"
            )
        lines.append("  Soft constraints:")
        for k in Weights.SOFT_KEYS:
            w = getattr(self.weights, k)
            lines.append(
                f"    {k:<22} {self.counts[k]:>4} x {w:<4} = {self.counts[k] * w:>6}"
            )
        lines.append(
            f"  -> total pelanggaran keras = {self.hard_violations}"
            f" ({'FEASIBLE' if self.feasible else 'TIDAK FEASIBLE'}),"
            f" penalti lunak = {self.soft_penalty}"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Instansi masalah
# --------------------------------------------------------------------------- #

State = List[Tuple[int, int]]
"""State = daftar pasangan (slot_waktu, ruangan) untuk setiap mata kuliah.

Panjang state selalu sama dengan jumlah mata kuliah, sehingga setiap state
merupakan *jadwal utuh* — ciri khas complete-state formulation.
"""


class TimetableProblem:
    """Instansi persoalan penjadwalan mata kuliah."""

    def __init__(
        self,
        courses: Sequence[Course],
        rooms: Sequence[Room],
        lecturers: Sequence[str],
        groups: Sequence[str],
        unavailable: Dict[int, Sequence[int]] | None = None,
        weights: Weights | None = None,
        name: str = "instance",
    ) -> None:
        self.name = name
        self.courses = list(courses)
        self.rooms = list(rooms)
        self.lecturers = list(lecturers)
        self.groups = list(groups)
        self.weights = weights or Weights()
        self.n = len(self.courses)
        self.n_rooms = len(self.rooms)
        self.n_slots = N_SLOTS

        # unavailable[l] = daftar slot saat dosen l tidak bersedia mengajar
        self.unavailable: Dict[int, set] = {
            l: set(v) for l, v in (unavailable or {}).items()
        }

        # --- pra-komputasi untuk mempercepat evaluasi ---------------------- #
        self._cap = [r.capacity for r in self.rooms]
        self._is_lab_room = [r.kind == "LAB" for r in self.rooms]
        self._lect = [c.lecturer for c in self.courses]
        self._stu = [c.students for c in self.courses]
        self._needs_lab = [c.needs_lab for c in self.courses]
        self._course_groups = [c.groups for c in self.courses]
        self._unavail_flat = {
            (l, t) for l, slots in self.unavailable.items() for t in slots
        }

    # -- konstruksi ------------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lecturers": self.lecturers,
            "groups": self.groups,
            "rooms": [
                {"id": r.id, "name": r.name, "capacity": r.capacity, "kind": r.kind}
                for r in self.rooms
            ],
            "courses": [
                {
                    "id": c.id,
                    "code": c.code,
                    "name": c.name,
                    "lecturer": c.lecturer,
                    "groups": list(c.groups),
                    "students": c.students,
                    "needs_lab": c.needs_lab,
                }
                for c in self.courses
            ],
            "unavailable": {str(k): sorted(v) for k, v in self.unavailable.items()},
            "weights": {
                k: getattr(self.weights, k)
                for k in Weights.HARD_KEYS + Weights.SOFT_KEYS
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "TimetableProblem":
        rooms = [Room(**r) for r in d["rooms"]]
        courses = [
            Course(
                id=c["id"],
                code=c["code"],
                name=c["name"],
                lecturer=c["lecturer"],
                groups=tuple(c["groups"]),
                students=c["students"],
                needs_lab=c["needs_lab"],
            )
            for c in d["courses"]
        ]
        unav = {int(k): v for k, v in d.get("unavailable", {}).items()}
        return cls(
            courses=courses,
            rooms=rooms,
            lecturers=d["lecturers"],
            groups=d["groups"],
            unavailable=unav,
            weights=Weights(**d.get("weights", {})),
            name=d.get("name", "instance"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TimetableProblem":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- initial state ---------------------------------------------------- #

    def random_state(self, rng: random.Random) -> State:
        """Initial state dibangkitkan **secara random dan tanpa perbaikan apa pun**.

        Setiap mata kuliah memperoleh pasangan (slot, ruang) yang diundi
        seragam dari seluruh domain. Konsekuensinya state awal hampir selalu
        melanggar banyak hard constraint — inilah titik berangkat local search.
        """
        return [
            (rng.randrange(self.n_slots), rng.randrange(self.n_rooms))
            for _ in range(self.n)
        ]

    # -- objective function ------------------------------------------------ #

    def evaluate(self, state: State) -> Cost:
        """Hitung f(X) beserta rincian tiap komponen. Semakin kecil semakin baik."""
        c = self.counts(state)
        w = self.weights
        total = sum(c[k] * getattr(w, k) for k in Weights.HARD_KEYS + Weights.SOFT_KEYS)
        return Cost(total=total, counts=c, weights=w)

    def objective(self, state: State) -> int:
        """Versi ringan ``evaluate`` yang hanya mengembalikan nilai f(X)."""
        c = self.counts(state)
        w = self.weights
        return sum(c[k] * getattr(w, k) for k in Weights.HARD_KEYS + Weights.SOFT_KEYS)

    def counts(self, state: State) -> Dict[str, int]:
        """Cacah pelanggaran mentah tiap komponen (belum dikalikan bobot)."""
        n = self.n
        cap = self._cap
        is_lab_room = self._is_lab_room
        lect = self._lect
        stu = self._stu
        needs_lab = self._needs_lab
        cgroups = self._course_groups
        unavail = self._unavail_flat

        room_slot: Dict[Tuple[int, int], int] = {}
        lect_slot: Dict[Tuple[int, int], int] = {}
        group_slot: Dict[Tuple[int, int], int] = {}
        lect_day: Dict[Tuple[int, int], List[int]] = {}
        group_day: Dict[Tuple[int, int], List[int]] = {}

        room_clash = 0
        lect_clash = 0
        group_clash = 0
        capacity = 0
        room_type = 0
        lect_unavail = 0
        late = 0
        waste = 0
        lab_misuse = 0

        for i in range(n):
            t, r = state[i]
            d = t // N_PERIODS
            p = t - d * N_PERIODS

            # H1 — dua kelas di ruang & slot yang sama
            k = (t, r)
            prev = room_slot.get(k, 0)
            room_clash += prev
            room_slot[k] = prev + 1

            # H2 — dosen mengajar dua kelas pada slot yang sama
            k = (t, lect[i])
            prev = lect_slot.get(k, 0)
            lect_clash += prev
            lect_slot[k] = prev + 1

            # H3 — satu kelompok mahasiswa punya dua kuliah pada slot yang sama
            for g in cgroups[i]:
                k = (t, g)
                prev = group_slot.get(k, 0)
                group_clash += prev
                group_slot[k] = prev + 1
                group_day.setdefault((g, d), []).append(p)

            # H4 — kapasitas ruangan
            free = cap[r] - stu[i]
            if free < 0:
                capacity += 1
            else:
                waste += free // 10  # S5 — kursi menganggur (per kelipatan 10)

            # H5 — praktikum wajib di ruang lab
            if needs_lab[i] and not is_lab_room[r]:
                room_type += 1
            # S6 — kuliah teori memakai ruang lab (lab terpakai sia-sia)
            elif (not needs_lab[i]) and is_lab_room[r]:
                lab_misuse += 1

            # S1 — dosen dijadwalkan pada slot yang ia nyatakan tidak bersedia
            if (lect[i], t) in unavail:
                lect_unavail += 1

            # S2 — kelas pada periode terakhir hari itu
            if p == N_PERIODS - 1:
                late += 1

            # akumulasi beban harian dosen (S3, S7)
            lect_day.setdefault((lect[i], d), []).append(p)

        # S3 — dosen mengajar lebih dari 2 kelas pada hari yang sama
        overload = sum(len(v) - 2 for v in lect_day.values() if len(v) > 2)

        # S4 — kelompok mahasiswa kuliah lebih dari 3 sesi pada hari yang sama
        group_overload = sum(len(v) - 3 for v in group_day.values() if len(v) > 3)

        # S6 — jam kosong menganggur di tengah jadwal harian sebuah kelompok
        gap = 0
        for periods in group_day.values():
            if len(periods) > 1:
                gap += (max(periods) - min(periods) + 1) - len(periods)

        # S7 — jam kosong menganggur di tengah jadwal harian seorang dosen
        lect_gap = 0
        for periods in lect_day.values():
            if len(periods) > 1:
                lect_gap += (max(periods) - min(periods) + 1) - len(periods)

        return {
            "room_clash": room_clash,
            "lecturer_clash": lect_clash,
            "group_clash": group_clash,
            "capacity": capacity,
            "room_type": room_type,
            "lecturer_unavailable": lect_unavail,
            "late_slot": late,
            "lecturer_overload": overload,
            "group_overload": group_overload,
            "group_gap": gap,
            "lecturer_gap": lect_gap,
            "room_waste": waste,
            "lab_misuse": lab_misuse,
        }

    # -- successor / neighbor function ------------------------------------- #

    def neighbors(self, state: State) -> Iterator[Tuple[State, tuple]]:
        """Successor function: seluruh tetangga state pada satu iterasi.

        Tiga jenis *move* yang diperbolehkan:

        1. ``("time", i, t)``  — pindahkan mata kuliah ``i`` ke slot waktu ``t``
           (ruangan tetap).
        2. ``("room", i, r)``  — pindahkan mata kuliah ``i`` ke ruangan ``r``
           (slot waktu tetap).
        3. ``("swap", i, j)``  — tukar pasangan (slot, ruang) mata kuliah ``i``
           dan ``j``.

        Ukuran neighborhood = ``n*(|T|-1) + n*(|R|-1) + n*(n-1)/2``.
        """
        n = self.n
        for i in range(n):
            t_i, r_i = state[i]
            for t in range(self.n_slots):
                if t != t_i:
                    nxt = list(state)
                    nxt[i] = (t, r_i)
                    yield nxt, ("time", i, t)
            for r in range(self.n_rooms):
                if r != r_i:
                    nxt = list(state)
                    nxt[i] = (t_i, r)
                    yield nxt, ("room", i, r)
        for i in range(n):
            for j in range(i + 1, n):
                if state[i] != state[j]:
                    nxt = list(state)
                    nxt[i], nxt[j] = state[j], state[i]
                    yield nxt, ("swap", i, j)

    def neighborhood_size(self) -> int:
        n = self.n
        return n * (self.n_slots - 1) + n * (self.n_rooms - 1) + n * (n - 1) // 2

    def random_neighbor(self, state: State, rng: random.Random) -> Tuple[State, tuple]:
        """Satu tetangga acak — dipakai Stochastic HC, Simulated Annealing, mutasi GA."""
        kind = rng.random()
        nxt = list(state)
        if kind < 0.45:
            i = rng.randrange(self.n)
            t_i, r_i = state[i]
            t = rng.randrange(self.n_slots)
            while t == t_i and self.n_slots > 1:
                t = rng.randrange(self.n_slots)
            nxt[i] = (t, r_i)
            return nxt, ("time", i, t)
        if kind < 0.80:
            i = rng.randrange(self.n)
            t_i, r_i = state[i]
            r = rng.randrange(self.n_rooms)
            while r == r_i and self.n_rooms > 1:
                r = rng.randrange(self.n_rooms)
            nxt[i] = (t_i, r)
            return nxt, ("room", i, r)
        i = rng.randrange(self.n)
        j = rng.randrange(self.n)
        while j == i and self.n > 1:
            j = rng.randrange(self.n)
        nxt[i], nxt[j] = state[j], state[i]
        return nxt, ("swap", i, j)

    def apply_move(self, state: State, move: tuple) -> State:
        """Terapkan sebuah move pada state (mengembalikan state baru)."""
        nxt = list(state)
        kind = move[0]
        if kind == "time":
            _, i, t = move
            nxt[i] = (t, state[i][1])
        elif kind == "room":
            _, i, r = move
            nxt[i] = (state[i][0], r)
        elif kind == "swap":
            _, i, j = move
            nxt[i], nxt[j] = state[j], state[i]
        else:  # pragma: no cover
            raise ValueError(f"move tidak dikenal: {move}")
        return nxt

    def describe_move(self, move: tuple) -> str:
        kind = move[0]
        if kind == "time":
            _, i, t = move
            return f"time  {self.courses[i].code} -> {slot_name(t)}"
        if kind == "room":
            _, i, r = move
            return f"room  {self.courses[i].code} -> {self.rooms[r].name}"
        _, i, j = move
        return f"swap  {self.courses[i].code} <-> {self.courses[j].code}"


# --------------------------------------------------------------------------- #
# Instansi bawaan
# --------------------------------------------------------------------------- #

_LECTURERS = [
    "Dr. Adiwijaya",
    "Dr. Bagus Priambodo",
    "Dr. Citra Handayani",
    "Dr. Dimas Kurniawan",
    "Dr. Elvira Setiawan",
    "Dr. Fajar Nugroho",
    "Dr. Gita Rahmawati",
    "Dr. Hadi Santosa",
    "Dr. Indira Puspita",
    "Dr. Joko Susilo",
]

_GROUPS = [
    "IF-23-K01",
    "IF-23-K02",
    "IF-24-K01",
    "IF-24-K02",
    "STI-23-K01",
    "STI-24-K01",
    "EL-23-K01",
    "EL-24-K01",
]

_ROOMS = [
    ("7502", 120, "KULIAH"),
    ("7602", 100, "KULIAH"),
    ("7603", 100, "KULIAH"),
    ("9009", 80, "KULIAH"),
    ("9010", 80, "KULIAH"),
    ("7610", 60, "KULIAH"),
    ("7611", 60, "KULIAH"),
    ("7612", 40, "KULIAH"),
    ("Lab-Dasar", 60, "LAB"),
    ("Lab-IRK", 40, "LAB"),
    ("Lab-Grafika", 40, "LAB"),
]

# (kode, nama, dosen, kelompok, jumlah mahasiswa, butuh lab)
_COURSES = [
    ("IF2110", "Algoritma dan Struktur Data", 0, (2, 3), 96, False),
    ("IF2110-P", "Praktikum Algoritma dan Struktur Data", 0, (2,), 48, True),
    ("IF2120", "Matematika Diskrit", 1, (2, 3), 96, False),
    ("IF2123", "Aljabar Linier dan Geometri", 1, (2, 3), 92, False),
    ("IF2124", "Teori Bahasa Formal dan Otomata", 2, (2, 3), 90, False),
    ("IF2130", "Organisasi dan Arsitektur Komputer", 3, (2, 3), 88, False),
    ("IF2130-P", "Praktikum Organisasi Komputer", 3, (3,), 44, True),
    ("IF2211", "Strategi Algoritma", 2, (0, 1), 94, False),
    ("IF2211-P", "Praktikum Strategi Algoritma", 2, (0,), 47, True),
    ("IF2210", "Pemrograman Berorientasi Objek", 4, (0, 1), 94, False),
    ("IF2210-P", "Praktikum PBO", 4, (1,), 47, True),
    ("IF2220", "Probabilitas dan Statistika", 5, (0, 1), 90, False),
    ("IF2230", "Sistem Operasi", 3, (0, 1), 92, False),
    ("IF2240", "Basis Data", 6, (0, 1), 92, False),
    ("IF2240-P", "Praktikum Basis Data", 6, (0,), 46, True),
    ("IF3110", "Pengembangan Aplikasi Web", 7, (0,), 52, True),
    ("IF3130", "Jaringan Komputer", 3, (1,), 50, False),
    ("IF3140", "Manajemen Basis Data", 6, (1,), 48, False),
    ("IF3170", "Inteligensi Buatan", 8, (0, 1), 88, False),
    ("IF3170-P", "Praktikum Inteligensi Buatan", 8, (1,), 44, True),
    ("IF3210", "Rekayasa Perangkat Lunak", 7, (0,), 54, False),
    ("IF3230", "Sistem Paralel dan Terdistribusi", 9, (1,), 46, False),
    ("IF3260", "Grafika Komputer", 9, (0,), 38, True),
    ("STI2110", "Sistem dan Teknologi Informasi", 4, (4, 5), 70, False),
    ("STI2120", "Sistem Informasi Perusahaan", 5, (4,), 36, False),
    ("STI2130", "Analisis Proses Bisnis", 6, (4, 5), 68, False),
    ("STI2210", "Manajemen Proyek TI", 7, (4,), 34, False),
    ("STI2220", "Interaksi Manusia dan Komputer", 8, (5,), 34, False),
    ("STI2230", "Keamanan Informasi", 9, (4,), 32, False),
    ("EL2140", "Sistem Digital", 3, (6, 7), 76, False),
    ("EL2140-P", "Praktikum Sistem Digital", 3, (6,), 38, True),
    ("EL2240", "Sinyal dan Sistem", 1, (6, 7), 74, False),
    ("EL3140", "Sistem Mikroprosesor", 9, (6,), 36, True),
    ("EL3240", "Elektronika Digital Lanjut", 5, (7,), 34, False),
    ("MA2071", "Matematika Teknik", 1, (6, 7), 72, False),
    ("KU2071", "Pancasila dan Kewarganegaraan", 0, (2, 3, 4), 110, False),
]

# slot yang dinyatakan tidak tersedia oleh masing-masing dosen (soft constraint)
_UNAVAILABLE = {
    0: [0, 1, 20],
    1: [4, 9, 14],
    2: [5, 6, 24],
    3: [10, 11, 12],
    4: [0, 15, 16],
    5: [19, 23, 24],
    6: [2, 3, 22],
    7: [7, 8, 13],
    8: [17, 18, 21],
    9: [0, 5, 10],
}


def default_problem() -> TimetableProblem:
    """Instansi bawaan: 36 kelas, 11 ruangan, 10 dosen, 8 kelompok, 25 slot."""
    rooms = [
        Room(id=i, name=n, capacity=c, kind=k)
        for i, (n, c, k) in enumerate(_ROOMS)
    ]
    courses = [
        Course(
            id=i,
            code=code,
            name=name,
            lecturer=lect,
            groups=tuple(groups),
            students=stu,
            needs_lab=lab,
        )
        for i, (code, name, lect, groups, stu, lab) in enumerate(_COURSES)
    ]
    return TimetableProblem(
        courses=courses,
        rooms=rooms,
        lecturers=_LECTURERS,
        groups=_GROUPS,
        unavailable=_UNAVAILABLE,
        weights=Weights(),
        name="STEI-ITB (36 kelas / 11 ruang / 25 slot)",
    )
