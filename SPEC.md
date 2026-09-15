SPEC.md のたたき台です。
- 何を入力するか
  - n個のparquetファイル
  - 1個のparquetファイルの中身は以下のとおり
    - Time列 (t)
    - Step列 (s)
    - 各波長列 (w)
    - 各セルはTime時点での各波長のスペクトル強度が格納されてる
- Flatten-PCAで行う処理
  - (optional) 前処理。t方向にsmoothingする
  - (optional) 前処理。w方向にsmoothingする
  - 各parquetファイルを列方向に展開し、1行にする
    - 列名は`f"{w}_{s}_{t}"`
    - のちのjoin操作でユニークキーとして使うため、filename列にpath.stemを格納しておく
  - n数のparquetファイルで同じ処理をして、concatして行数nのdf: pl.DataFrame を得る
  - dfをpcaかけてdf_pca: pl.DataFrame を得る
- 何を出力するか
  - dfとdf_pcaをjoinしたpl.DataFrame
- Python API、CLI、Notebookのどれを成果物とするか
  - Python API

## v1仕様

### スコープ

- v1では、複数のParquetファイルをflattenし、同じデータ集合にPCAをfitして得た主成分スコアを結合した`pl.DataFrame`を返す。
- v1の公開APIは、学習済みPCAモデルの保存や未知データへのtransformを扱わない。
- t方向およびw方向のsmoothingは、手法、窓幅、境界処理を別途定義する必要があるためv1の対象外とする。
- CLIおよびNotebookは成果物に含めない。

### 公開API

公開関数は次のシグネチャとする。

```python
def flatten_pca(
    paths: Sequence[str | Path],
    *,
    n_components: int,
) -> pl.DataFrame:
    ...
```

- `paths`は1個以上のParquetファイルへのパスとする。
- `n_components`は1以上かつ`min(n_samples, n_features)`以下の整数とする。
- 戻り値はflatten結果とPCAスコアを`filename`で結合した`pl.DataFrame`とする。

### 入力要件

- すべての入力ファイルに`Time`列と`Step`列が存在しなければならない。
- `Time`列と`Step`列は数値型でなければならない。
- `Time`列と`Step`列以外の列名は、有限な数値の波長として解釈できなければならない。
- スペクトル強度は数値型でなければならない。
- null、NaNおよび無限大を含む入力は受け付けない。
- 各ファイル内で`(Time, Step)`の組み合わせは一意でなければならない。
- すべてのファイルは、同一の波長集合および同一の`(Time, Step)`集合を持たなければならない。
- `path.stem`を`filename`として使用するため、入力ファイル間で`path.stem`は一意でなければならない。
- 入力パスが空、ファイルが存在しない、または上記の要件を満たさない場合は例外を送出する。
  - 存在しないファイルには`FileNotFoundError`を使用する。
  - 値、スキーマおよび引数の不正には`ValueError`を使用する。

### flatten仕様

- 入力ファイルの処理順は、正規化したパスの文字列表現による昇順とする。
- flatten前に、波長、`Step`、`Time`をそれぞれ数値として昇順に並べる。
- flatten後の特徴量列は、波長、`Step`、`Time`の順を優先して並べる。
- 特徴量列名は`f"{w}_{s}_{t}"`の形式とする。
- `w`、`s`、`t`は、不要な末尾の0を除いた最大15桁の有効数字で表現する。
- 正規化後の特徴量列名が重複する場合は`ValueError`を送出する。
- flatten結果`df`の列順は、`filename`、flattenした特徴量列の順とする。

### PCA仕様

- PCAの入力には`filename`を除くすべてのflatten特徴量列を使用する。
- `sklearn.decomposition.PCA`を使用する。
- PCA内部の平均中心化は行うが、分散による標準化は行わない。
- 再現可能な計算経路にするため、`svd_solver="full"`を使用する。
- PCAスコアの列名は`PC1`、`PC2`、...、`PC{n_components}`とする。
- `df_pca`は`filename`とPCAスコア列を持つ。
- PCA成分の符号は一意に定まらないため、テストでは主成分やスコアの符号そのものを固定値と単純比較しない。

### 出力要件

- 戻り値の行数は入力ファイル数nと一致しなければならない。
- 戻り値の各行は、`filename`によって元の入力ファイルを一意に識別できなければならない。
- 列順は、`filename`、flatten特徴量列、PCAスコア列の順とする。
- 入力パスの指定順に依存せず、同じファイル集合から同じ行順および列順を得られなければならない。

### 完了条件

- 公開APIと公開される関数・クラスには型ヒントとNumPy形式のdocstringがある。
- 正常系、入力検証、並び順、flatten結果、PCA結果を対象とした自動テストがある。
- PCAのテストは、符号反転を許容しながら分散、部分空間または再構成結果を検証する。
- `uv run -m pytest`が成功する。
- `TASKS.json`に定義されたすべてのタスクが、対応する要件とテストを満たして完了している。
