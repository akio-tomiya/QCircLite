import math

import numpy as np
import pytest

from qcirclite import Circuit, State

S2 = 1 / math.sqrt(2)


def vec(c):
    return c.run().vector


# ---- 基本と、ビット列の順番 ------------------------------------------------

def test_initial_state_is_all_zero():
    assert np.allclose(vec(Circuit(2)), [1, 0, 0, 0])


def test_left_digit_is_qubit_zero():
    # ノートの決め方：左の桁が0番（1個目）の量子ビット
    assert Circuit(2).x[0].run().probabilities() == {"10": 1.0}
    assert Circuit(2).x[1].run().probabilities() == {"01": 1.0}
    assert np.allclose(vec(Circuit(2).x[0]), [0, 0, 1, 0])


def test_qubit_count_grows_automatically():
    c = Circuit().h[0].cx[0, 2]
    assert c.n_qubits == 3


# ---- 1量子ビット（第3・4回の手計算） ------------------------------------------

def test_hadamard_twice_is_identity():
    assert np.allclose(vec(Circuit().h[0].h[0]), [1, 0])


def test_hzh_flips_zero_to_one():
    assert np.allclose(vec(Circuit().h[0].z[0].h[0]), [0, 1])


def test_hxh_is_z_on_plus_state():
    # HXH|+> = Z|+> = |->
    assert np.allclose(vec(Circuit().h[0].h[0].x[0].h[0]), [S2, -S2])


def test_rx_pi_is_minus_i_x():
    assert np.allclose(vec(Circuit().rx(math.pi)[0]), [0, -1j])


def test_rz_on_plus_goes_to_y_axis():
    # Rz(π/2)|+> は全体位相を除いて (|0> + i|1>)/√2
    v = vec(Circuit().h[0].rz(math.pi / 2)[0])
    v = v / v[0] * abs(v[0])
    assert np.allclose(v, [S2, 1j * S2])


# ---- 2量子ビット（第5回の手計算） ---------------------------------------------

def test_bell_state_amplitudes():
    assert np.allclose(vec(Circuit().h[0].cx[0, 1]), [S2, 0, 0, S2])


def test_kronecker_zero_one():
    assert np.allclose(vec(Circuit(2).x[1]), [0, 1, 0, 0])


def test_cnot_copies_basis_but_not_plus():
    assert Circuit().x[0].cx[0, 1].run().probabilities() == {"11": 1.0}
    # |+>|0> に CNOT を掛けてもコピー |++> にはならない
    assert not np.allclose(vec(Circuit().h[0].cx[0, 1]), [0.5, 0.5, 0.5, 0.5])


def test_toffoli_truth_table():
    for a in (0, 1):
        for b in (0, 1):
            c = Circuit(3)
            if a:
                c.x[0]
            if b:
                c.x[1]
            c.ccx[0, 1, 2]
            assert c.run().probabilities() == {f"{a}{b}{a & b}": 1.0}


# ---- 簡単なアルゴリズム（第6回） -----------------------------------------------

DEUTSCH_ORACLES = {
    "const0": lambda c: c,
    "const1": lambda c: c.x[1],
    "identity": lambda c: c.cx[0, 1],
    "negation": lambda c: c.cx[0, 1].x[1],
}


@pytest.mark.parametrize("name", DEUTSCH_ORACLES)
def test_deutsch_decides_constant_or_balanced(name):
    c = Circuit(2).x[1].h[0].h[1]
    DEUTSCH_ORACLES[name](c)
    c.h[0].m[0]
    expected = "0" if name.startswith("const") else "1"
    assert c.run().probabilities(qubits=[0]) == pytest.approx({expected: 1.0})


@pytest.mark.parametrize("marked", ["00", "01", "10", "11"])
def test_two_qubit_grover_finds_marked_item(marked):
    flips = [q for q, bit in enumerate(marked) if bit == "0"]

    def oracle(c):
        for q in flips:
            c.x[q]
        c.cz[0, 1]
        for q in flips:
            c.x[q]
        return c

    c = Circuit(2).h[0].h[1]
    oracle(c)
    c.h[0].h[1].x[0].x[1].cz[0, 1].x[0].x[1].h[0].h[1]
    probs = c.run().probabilities()
    assert probs[marked] == pytest.approx(1.0)


# ---- 測定と表示 ------------------------------------------------------------

def test_counts_are_reproducible_and_sum_to_shots():
    c = Circuit().h[0].cx[0, 1].m[:]
    a = c.run(shots=1000, seed=1)
    b = c.run(shots=1000, seed=1)
    assert a == b
    assert sum(a.values()) == 1000
    assert set(a) <= {"00", "11"}


def test_measure_subset_gives_marginal():
    c = Circuit().h[0].x[1].m[1]
    assert c.run(shots=50, seed=0) == {"1": 50}


def test_gate_after_measurement_is_rejected():
    with pytest.raises(ValueError):
        Circuit().h[0].m[0].x[0]


def test_slice_needs_known_size():
    with pytest.raises(ValueError):
        Circuit().h[:]
    assert Circuit(3).h[:].n_qubits == 3


def test_state_repr_is_ket_form():
    assert repr(Circuit().h[0].cx[0, 1].run()) == "0.707|00⟩ + 0.707|11⟩"
    assert repr(Circuit().x[0].h[0].run()) == "0.707|0⟩ - 0.707|1⟩"
    assert repr(Circuit().h[0].s[0].run()) == "0.707|0⟩ + 0.707j|1⟩"


def test_text_drawing_has_gate_symbols():
    text = repr(Circuit().h[0].cx[0, 1].m[:])
    assert "H" in text and "●" in text and "⊕" in text and "M" in text
    assert text.splitlines()[0].startswith("q0:")


def test_parametric_gate_needs_angle():
    with pytest.raises(TypeError):
        Circuit().rz[0]


def test_matplotlib_outputs():
    import matplotlib
    matplotlib.use("Agg")
    c = Circuit().h[0].cx[0, 1].m[:]
    assert c.draw("mpl") is not None
    assert c.plot(shots=100, seed=0) is not None


def test_state_accepts_bitstring_index():
    s = Circuit().h[0].cx[0, 1].run()
    assert isinstance(s, State)
    assert s["11"] == pytest.approx(S2)


def test_probabilities_are_rounded_for_display():
    assert Circuit(1).h[0].run().probabilities() == {"0": 0.5, "1": 0.5}
