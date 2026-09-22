"""QCircLite: 講義用の小さな量子回路シミュレータ。

依存は numpy だけです（図示とグラフには matplotlib を使います）。
1ファイルなので、Google Colab では次のように取り寄せて使えます。

    !curl -O <配布URL>/qcirclite.py
    from qcirclite import Circuit

    c = Circuit().h[0].cx[0, 1]
    c                          # 回路図（テキスト）
    c.run()                    # 状態ベクトル 0.707|00⟩ + 0.707|11⟩
    c.m[:].run(shots=1000)     # 測定結果の回数
    c.plot(shots=1000)         # 回数の棒グラフ（理論上の回数と並べる）

ビット列の順番：左の桁が0番（1個目）の量子ビットです。
たとえば Circuit(2).x[0] の状態は |10⟩ です。
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

__version__ = "0.1.0"
__all__ = ["Circuit", "State", "Counts", "plot_counts", "__version__"]

_EPS = 1e-12
_SQRT2_INV = 1 / math.sqrt(2)


# ---------------------------------------------------------------------------
# ゲートの行列
# ---------------------------------------------------------------------------

def _rx(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta):
    return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex)


def _phase(theta):
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


_FIXED = {
    "i": np.eye(2, dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "h": np.array([[1, 1], [1, -1]], dtype=complex) * _SQRT2_INV,
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(0.25j * math.pi)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-0.25j * math.pi)]], dtype=complex),
    "cx": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    "cz": np.diag([1, 1, 1, -1]).astype(complex),
    "swap": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
}
_ccx = np.eye(8, dtype=complex)
_ccx[[6, 7]] = _ccx[[7, 6]]
_FIXED["ccx"] = _ccx

_PARAMETRIC = {"rx": _rx, "ry": _ry, "rz": _rz, "phase": _phase}

# ゲート名 -> 作用する量子ビットの数
_ARITY = {name: int(round(math.log2(m.shape[0]))) for name, m in _FIXED.items()}
_ARITY.update({name: 1 for name in _PARAMETRIC})
_ARITY["m"] = 1

# 別名（blueqat や教科書での呼び方）
_ALIASES = {"cnot": "cx", "toffoli": "ccx", "measure": "m", "p": "phase"}

_LABELS = {"sdg": "S†", "tdg": "T†"}


def _gate_matrix(name, params):
    if name in _PARAMETRIC:
        return _PARAMETRIC[name](*params)
    return _FIXED[name]


def _label(name, params):
    if name in _PARAMETRIC:
        base = {"rx": "Rx", "ry": "Ry", "rz": "Rz", "phase": "P"}[name]
        return f"{base}({_short(params[0])})"
    return _LABELS.get(name, name.upper())


def _short(x):
    for num, text in ((math.pi, "π"), (math.pi / 2, "π/2"), (math.pi / 4, "π/4")):
        if abs(x - num) < 1e-9:
            return text
        if abs(x + num) < 1e-9:
            return "-" + text
    return f"{x:.3g}"


# ---------------------------------------------------------------------------
# 回路
# ---------------------------------------------------------------------------

class _GateIndexer:
    """c.h[0] や c.rz(θ)[0] の [ ] を受け取る小さな窓口。"""

    def __init__(self, circuit, name, params=()):
        self._circuit = circuit
        self._name = name
        self._params = tuple(params)

    def __call__(self, *params):
        if self._name not in _PARAMETRIC:
            raise TypeError(f"{self._name} は角度を取らないゲートです。c.{self._name}[0] のように書いてください。")
        if len(params) != 1:
            raise TypeError(f"{self._name} には角度を1つ与えてください。例：c.{self._name}(math.pi / 2)[0]")
        return _GateIndexer(self._circuit, self._name, params)

    def __getitem__(self, key):
        if self._name in _PARAMETRIC and not self._params:
            raise TypeError(f"{self._name} には角度が必要です。例：c.{self._name}(math.pi / 2)[0]")
        return self._circuit._add(self._name, key, self._params)


class Circuit:
    """量子回路。ゲートを c.h[0].cx[0, 1] のように続けて書く。

    Circuit() と書くと、使った量子ビットの番号に合わせて数が自動で増えます。
    Circuit(3) のように最初から数を決めることもできます（[:] で全部を指すときに便利です）。
    """

    def __init__(self, n_qubits=0):
        if n_qubits < 0:
            raise ValueError("量子ビットの数は0以上にしてください。")
        self.n_qubits = int(n_qubits)
        self.ops = []           # (名前, 量子ビットのタプル, パラメータのタプル)
        self.measured = []      # 測定する量子ビット（番号順）

    # c.h, c.cx, c.rz(θ), c.m などの窓口
    def __getattr__(self, name):
        name = _ALIASES.get(name, name)
        if name in _ARITY:
            return _GateIndexer(self, name)
        raise AttributeError(f"'{name}' というゲートはありません。使えるゲート：{', '.join(sorted(_ARITY))}")

    def _targets(self, key, arity):
        """[ ] の中身を、ゲートを置く量子ビットの組のリストへ直す。"""
        if arity == 1:
            if isinstance(key, tuple):
                groups = []
                for k in key:
                    groups.extend(self._targets(k, 1))
                return groups
            if isinstance(key, slice):
                if self.n_qubits == 0 and key.stop is None:
                    raise ValueError("量子ビットの数が決まっていないので [:] が使えません。Circuit(2) のように数を決めてください。")
                limit = self.n_qubits if key.stop is None else max(key.stop, self.n_qubits)
                return [(q,) for q in range(*key.indices(limit))]
            return [(self._index(key),)]
        if not isinstance(key, tuple) or len(key) != arity:
            raise ValueError(f"このゲートは量子ビットを{arity}個指定します。例：[0, 1]")
        qubits = tuple(self._index(k) for k in key)
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"同じ量子ビットを2回指定しています：{qubits}")
        return [qubits]

    @staticmethod
    def _index(k):
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError(f"量子ビットの番号は0以上の整数で書いてください：{k!r}")
        return int(k)

    def _add(self, name, key, params):
        groups = self._targets(key, _ARITY[name])
        for qubits in groups:
            self.n_qubits = max(self.n_qubits, max(qubits) + 1)
            if name == "m":
                if qubits[0] not in self.measured:
                    self.measured.append(qubits[0])
                    self.measured.sort()
                continue
            if self.measured:
                raise ValueError("測定（m）の後にゲートは置けません。測定は回路の最後に書いてください。")
            self.ops.append((name, qubits, params))
        return self

    def copy(self):
        c = Circuit(self.n_qubits)
        c.ops = list(self.ops)
        c.measured = list(self.measured)
        return c

    # ---- 実行 -------------------------------------------------------------

    def run(self, shots=None, seed=None):
        """shots を省くと状態ベクトル（State）を返す。

        shots=1000 のように回数を与えると、測定結果の回数（Counter）を返す。
        m で測定する量子ビットを指定していなければ、全部の量子ビットを測る。
        seed を与えると、同じ乱数で結果を再現できる。
        """
        if self.n_qubits == 0:
            raise ValueError("量子ビットが1個もありません。")
        psi = np.zeros(2 ** self.n_qubits, dtype=complex)
        psi[0] = 1
        for name, qubits, params in self.ops:
            psi = _apply(psi, _gate_matrix(name, params), qubits, self.n_qubits)
        state = State(psi, self.n_qubits)
        if shots is None:
            return state
        return state.sample(shots, qubits=self.measured or None, seed=seed)

    # ---- 表示 -------------------------------------------------------------

    def draw(self, output="text"):
        """回路図を描く。output="text"（文字）または "mpl"（matplotlib の図）。"""
        if output == "text":
            print(self._text())
            return None
        if output == "mpl":
            return _draw_mpl(self)
        raise ValueError('output は "text" か "mpl" にしてください。')

    def plot(self, shots=1000, seed=None, show_expected=True):
        """測定回数の棒グラフを描く。show_expected=True なら理論上の回数も並べる。"""
        counts = self.run(shots=shots, seed=seed)
        expected = None
        if show_expected:
            probs = self.run().probabilities(qubits=self.measured or None)
            expected = {k: v * shots for k, v in probs.items()}
        return plot_counts(counts, expected=expected)

    def __repr__(self):
        return self._text() if self.n_qubits else "Circuit()"

    def _columns(self):
        cols = [self._column(name, qubits, params) for name, qubits, params in self.ops]
        if self.measured:
            cols.append({q: ("M", False) for q in self.measured})
        return cols

    @staticmethod
    def _column(name, qubits, params):
        if name == "cx":
            return {qubits[0]: ("●", True), qubits[1]: ("⊕", True)}
        if name == "ccx":
            return {qubits[0]: ("●", True), qubits[1]: ("●", True), qubits[2]: ("⊕", True)}
        if name == "cz":
            return {qubits[0]: ("●", True), qubits[1]: ("●", True)}
        if name == "swap":
            return {qubits[0]: ("×", True), qubits[1]: ("×", True)}
        return {qubits[0]: (_label(name, params), False)}

    def _text(self):
        n = self.n_qubits
        width_name = len(f"q{n - 1}: ")
        lines = [f"q{q}: ".rjust(width_name) for q in range(n)]
        gaps = [" " * width_name for _ in range(n - 1)]
        for col in self._columns():
            w = max(len(label) for label, _ in col.values()) + 4
            lo, hi = min(col), max(col)
            vertical = any(v for _, v in col.values()) and len(col) > 1
            for q in range(n):
                if q in col:
                    label, _ = col[q]
                    cell = label.center(w, "─") if len(label) == 1 else f"─{label}─".center(w, "─")
                elif vertical and lo < q < hi:
                    cell = "┼".center(w, "─")
                else:
                    cell = "─" * w
                lines[q] += cell
            for g in range(n - 1):
                gaps[g] += ("│" if vertical and lo <= g < hi else " ").center(w)
        out = []
        for q in range(n):
            out.append(lines[q] + "─")
            if q < n - 1:
                out.append(gaps[q].rstrip())
        return "\n".join(out)


def _apply(psi, matrix, qubits, n):
    """状態ベクトル psi の、qubits で指定した量子ビットに行列を掛ける。"""
    k = len(qubits)
    tensor = psi.reshape([2] * n)
    gate = matrix.reshape([2] * (2 * k))
    tensor = np.tensordot(gate, tensor, axes=(list(range(k, 2 * k)), list(qubits)))
    tensor = np.moveaxis(tensor, list(range(k)), list(qubits))
    return tensor.reshape(-1)


# ---------------------------------------------------------------------------
# 状態ベクトル
# ---------------------------------------------------------------------------

class State:
    """状態ベクトル。0.707|00⟩ + 0.707|11⟩ のように表示する。

    state.vector で numpy の配列（成分の並びは |00⟩, |01⟩, |10⟩, |11⟩, ...）を取り出せる。
    """

    def __init__(self, vector, n_qubits):
        self.vector = np.asarray(vector, dtype=complex)
        self.n_qubits = n_qubits

    def __array__(self, dtype=None, copy=None):
        return self.vector if dtype is None else self.vector.astype(dtype)

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.vector[int(key, 2)]
        return self.vector[key]

    def _bits(self, index):
        return format(index, f"0{self.n_qubits}b")

    def amplitudes(self, digits=None):
        """0でない振幅を {'00': 振幅, ...} の辞書で返す。"""
        out = {}
        for i, a in enumerate(self.vector):
            if abs(a) > 1e-9:
                out[self._bits(i)] = a if digits is None else complex(round(a.real, digits), round(a.imag, digits))
        return out

    def probabilities(self, qubits=None):
        """測定確率を {'00': 確率, ...} の辞書で返す。qubits を与えると、その量子ビットだけの確率。"""
        probs = np.abs(self.vector) ** 2
        if qubits is not None:
            keep = sorted(qubits)
            tensor = probs.reshape([2] * self.n_qubits)
            others = tuple(q for q in range(self.n_qubits) if q not in keep)
            probs = tensor.sum(axis=others).reshape(-1) if others else tensor.reshape(-1)
            width = len(keep)
        else:
            width = self.n_qubits
        return {format(i, f"0{width}b"): float(p) for i, p in enumerate(probs) if p > 1e-12}

    def sample(self, shots, qubits=None, seed=None):
        """測定を shots 回くり返したときの結果の回数（Counter）を返す。"""
        probs = self.probabilities(qubits)
        keys = list(probs)
        p = np.array([probs[k] for k in keys])
        rng = np.random.default_rng(seed)
        draws = rng.choice(len(keys), size=int(shots), p=p / p.sum())
        counts = Counter(keys[i] for i in draws)
        return Counts(dict(sorted(counts.items())))

    def __repr__(self):
        terms = []
        for bits, a in self.amplitudes().items():
            terms.append((_coef(a), f"|{bits}⟩"))
        if not terms:
            return "0"
        text = ""
        for i, (coef, ket) in enumerate(terms):
            if i == 0:
                text = ("-" if coef.startswith("-") else "") + coef.lstrip("-") + ket
            else:
                sign = " - " if coef.startswith("-") else " + "
                text += sign + coef.lstrip("-") + ket
        return text

    def _repr_latex_(self):
        return "$" + repr(self).replace("⟩", r"\rangle").replace("|", r"\lvert ").replace("j", r"\,i") + "$"


class Counts(Counter):
    """測定結果の回数。Counter と同じように使え、ビット列の順に表示する。"""

    def __repr__(self):
        return "Counts({" + ", ".join(f"'{k}': {v}" for k, v in sorted(self.items())) + "})"


def _coef(a):
    re, im = a.real, a.imag
    if abs(im) < 1e-9:
        return _num(re)
    if abs(re) < 1e-9:
        return _num(im) + "j"
    return f"({_num(re)}{'+' if im >= 0 else '-'}{_num(abs(im))}j)"


def _num(x):
    return f"{x:.3f}".rstrip("0").rstrip(".") if abs(x - round(x)) > 1e-9 else str(int(round(x)))


# ---------------------------------------------------------------------------
# matplotlib での図示
# ---------------------------------------------------------------------------

def plot_counts(counts, expected=None, ax=None):
    """測定回数の棒グラフ。expected（理論上の回数の辞書）を与えると横に並べる。"""
    import matplotlib.pyplot as plt

    keys = sorted(set(counts) | set(expected or {}))
    x = np.arange(len(keys))
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(keys) + 2), 3.2))
    else:
        fig = ax.figure
    width = 0.4 if expected else 0.6
    ax.bar(x - (width / 2 if expected else 0), [counts.get(k, 0) for k in keys], width, label="measured")
    if expected:
        ax.bar(x + width / 2, [expected.get(k, 0) for k in keys], width, label="expected", alpha=0.6)
        ax.set_ylim(0, 1.25 * max(max(counts.values(), default=0), max(expected.values(), default=0)))
        ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_xlabel("outcome")
    ax.set_ylabel("counts")
    fig.tight_layout()
    return fig


def _draw_mpl(circuit):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    n = circuit.n_qubits
    cols = circuit._columns()
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * (len(cols) + 1), 0.4 + 0.8 * n))
    for q in range(n):
        ax.plot([0, len(cols) + 1], [-q, -q], color="black", lw=1, zorder=0)
        ax.text(-0.15, -q, f"q{q}", ha="right", va="center", fontsize=12)
    for i, col in enumerate(cols, start=1):
        if len(col) > 1 and any(connected for _, connected in col.values()):
            ys = [-q for q in col]
            ax.plot([i, i], [min(ys), max(ys)], color="black", lw=1, zorder=1)
        for q, (label, _) in col.items():
            y = -q
            if label == "●":
                ax.add_patch(Circle((i, y), 0.08, color="black", zorder=3))
            elif label == "⊕":
                ax.add_patch(Circle((i, y), 0.18, fill=False, color="black", lw=1.2, zorder=3))
                ax.plot([i - 0.18, i + 0.18], [y, y], color="black", lw=1.2, zorder=3)
                ax.plot([i, i], [y - 0.18, y + 0.18], color="black", lw=1.2, zorder=3)
            elif label == "×":
                d = 0.12
                ax.plot([i - d, i + d], [y - d, y + d], color="black", lw=1.5, zorder=3)
                ax.plot([i - d, i + d], [y + d, y - d], color="black", lw=1.5, zorder=3)
            else:
                w = max(0.5, 0.13 * len(label) + 0.25)
                ax.add_patch(Rectangle((i - w / 2, y - 0.25), w, 0.5, facecolor="white",
                                       edgecolor="black", zorder=3))
                ax.text(i, y, label, ha="center", va="center", fontsize=11, zorder=4)
    ax.set_xlim(-0.6, len(cols) + 1.2)
    ax.set_ylim(-(n - 1) - 0.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig
