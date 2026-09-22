# QCircLite

講義用の小さな量子回路シミュレータです。パッケージ名は `qcirclite` です。

- 1ファイル（`src/qcirclite.py`）で動きます。依存は numpy だけで、回路図とグラフには matplotlib を使います（どちらも Google Colab に最初から入っています）。
- 回路は `Circuit().h[0].cx[0, 1]` のように、ゲートを続けて書きます。
- 状態ベクトルを `0.707|00⟩ + 0.707|11⟩` のように表示するので、手計算と見比べられます。

## 使い方

```python
from qcirclite import Circuit

c = Circuit().h[0].cx[0, 1]   # ベル状態を作る回路
c                             # 回路図（テキスト）
c.draw("mpl")                 # 回路図（matplotlib）
c.run()                       # 状態ベクトル：0.707|00⟩ + 0.707|11⟩
c.run().vector                # numpy の配列（成分の並びは |00⟩, |01⟩, |10⟩, |11⟩）
c.m[:].run(shots=1000)        # 1000回測った結果：Counts({'00': 473, '11': 527})
c.plot(shots=1000, seed=0)    # 測定回数の棒グラフ（理論上の回数と並べる）
```

## ビット列の順番

**左の桁が0番（1個目）の量子ビット**です。講義ノートの決め方と同じです。

```python
Circuit(2).x[0].run()   # 1|10⟩
Circuit(2).x[1].run()   # 1|01⟩
```

（Qiskit などは右の桁を0番とするので、逆になります。）

## 書き方

| 書き方 | 意味 |
|---|---|
| `Circuit()` | 量子ビットの数は、使った番号に合わせて自動で増える |
| `Circuit(3)` | 量子ビットを3個用意する（`[:]` で全部を指すときに使う） |
| `c.h[0]`、`c.x[1]` | 1量子ビットのゲート |
| `c.h[0, 1]`、`c.h[:]` | 同じゲートを複数の量子ビットへ |
| `c.rz(math.pi / 2)[0]` | 角度を取るゲート |
| `c.cx[0, 1]` | CNOT（0番が制御、1番が標的） |
| `c.ccx[0, 1, 2]` | トフォリ（0番・1番が制御、2番が標的） |
| `c.m[0]`、`c.m[:]` | 測定する量子ビット（回路の最後に書く） |
| `c.run()` | 状態ベクトル（`State`） |
| `c.run(shots=1000, seed=0)` | 測定結果の回数（`Counts`）。`m` がなければ全部を測る |

使えるゲート：`i`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `rx(θ)`, `ry(θ)`, `rz(θ)`, `phase(θ)`, `cx`（別名 `cnot`）, `cz`, `swap`, `ccx`（別名 `toffoli`）, `m`（別名 `measure`）。

回転ゲートは $R_x(\theta)=e^{-i\theta X/2}$ などの定義です。

## Google Colab での使い方

版を固定して入れる（授業ではこちらを使う）：

```python
!pip -q install git+https://github.com/akio-tomiya/QCircLite@v0.1.0
from qcirclite import Circuit
```

最新版を入れる：

```python
!pip -q install git+https://github.com/akio-tomiya/QCircLite@main
```

ファイル1つだけを取り寄せてもよい：

```python
!curl -s -o qcirclite.py https://raw.githubusercontent.com/akio-tomiya/QCircLite/v0.1.0/src/qcirclite.py
```

どの版を使っているかは `import qcirclite; print(qcirclite.__version__)` で確かめられます。
一度 `import` した後で別の版を入れ直したときは、ランタイムを再起動してください。

## 開発

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

テストでは、講義の手計算（ベル状態、HZH、Deutsch のアルゴリズム、2量子ビットの Grover 探索など）と結果が一致することを確かめています。

記法は blueqat（Apache License 2.0）の `Circuit().h[0].cx[0, 1]` という書き方を参考にしました。コードは独自に書いたもので、blueqat のコードは含みません。

## ライセンス

MIT License（`LICENSE` を参照）。
