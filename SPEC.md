
## v1仕様

### スコープ

- v1では、複数のParquetファイルに前処理とflattenを適用し、その結果にPCAをfitする。
- `flatten_pca`はfit済みの`sklearn.decomposition.PCA`を返す。flatten結果およびPCAスコアを結合した`pl.DataFrame`は別の公開関数で取得する。
- v1の公開APIは、学習済みPCAモデルの永続化および未知データへのtransformを扱わない。
- t方向およびw方向のsmoothingと規格化は、任意の前処理としてv1の対象に含める。
- CLIおよびNotebookは成果物に含めない。

### 公開API

公開関数は次のシグネチャとする。`preprocess_and_flatten`は、従来の「前処理→flatten」部分を独立したAPIとして提供する。

```python
def preprocess_and_flatten(
    paths: Sequence[str | Path],
    *,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
    t_downsampling_stride: int = 1,
    w_downsampling_stride: int = 1,
) -> pl.LazyFrame:
    ...


def flatten_pca(
    paths: Sequence[str | Path],
    *,
    n_component: int | None = None,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
    t_downsampling_stride: int = 1,
    w_downsampling_stride: int = 1,
) -> PCA:
    ...


def append_pca_scores(
    pca: PCA,
    flattened: pl.LazyFrame,
) -> pl.LazyFrame:
    ...


def reshape_pca_components(
    pca: PCA,
    flattened: pl.LazyFrame,
) -> np.ndarray:
    ...
```

- `preprocess_and_flatten`の`paths`は1個以上のParquetファイルへのパスとする。
- `preprocess_and_flatten`は、`filename`列およびflatten特徴量列を持つ`pl.LazyFrame`を返す。列順と特徴量列名は「flatten仕様」に従う。
- `flatten_pca`の`paths`および前処理引数は`preprocess_and_flatten`と同じ意味・検証規則とする。内部で`preprocess_and_flatten`を呼び出し、その`filename`以外の列を特徴量としてPCAをfitして、fit済みの`PCA`を返す。
- `n_component`は1以上かつ`min(n_samples, n_features)`以下の整数、または`None`とする。`None`の場合はこの上限を使用する。
- `t_smoothing_window`はt方向smoothingの片側窓幅をTimeと同じ単位で指定し、`None`の場合は適用しない。
- `w_smoothing_window`はw方向smoothingの片側窓幅を波長と同じ単位で指定し、`None`の場合は適用しない。
- `t_normalization_range`はt方向規格化に用いる閉区間`(t1, t2)`を指定し、`None`の場合は適用しない。
- `w_normalization_range`はw方向規格化に用いる閉区間`(w1, w2)`を指定し、`None`の場合は適用しない。
- `t_downsampling_stride`はt方向の間引き間隔を指定する。`1`の場合は間引かない。
- `w_downsampling_stride`はw方向の間引き間隔を指定する。`1`の場合は間引かない。
- `append_pca_scores`は、fit済み`pca`とflatten済み`flattened`を受け取り、`filename`、flatten特徴量列およびPCAスコア列を持つ`pl.LazyFrame`を返す。スコア列は`pca-1`、`pca-2`、...、`pca-{n_component}`とする。
- `reshape_pca_components`は、fit済み`pca`とflatten済み`flattened`を受け取り、後述のshapeへ並べ替えた`pca.components_`を返す。

### LazyFrameとメモリ使用

- `preprocess_and_flatten`では、Parquetの読み込みに`pl.scan_parquet`を使用し、前処理、間引きおよびflattenの列演算を可能な限り`pl.LazyFrame`のまま構成する。
- 入力検証、共通のUnique配列の確定、PCAのfit・transformなど、結果値が必要な境界でのみ必要最小限のcollectを行う。前処理済みの全入力フレームを一括で`pl.DataFrame`へmaterializeしてはならない。
- `preprocess_and_flatten`、`append_pca_scores`は呼び出し側がcollectの時点と実行方法を選べるよう、常に`pl.LazyFrame`を返す。

### 入力要件

- すべての入力ファイルに`Time`列、`Step`列および`Sequence`列が存在しなければならない。
- `Time`列、`Step`列および`Sequence`列は数値型でなければならない。
- `Time`列、`Step`列および`Sequence`列以外を波長列として扱う。
- 波長列名は、有限な数値`v`を`f"{v:.1f}nm"`で表現した形式でなければならない。
- スペクトル強度は数値型でなければならない。
- null、NaNおよび無限大を含む入力は受け付けない。
- 各ファイル内で`(Time, Step, Sequence)`の組み合わせは一意でなければならない。
- すべてのファイルは、同一の波長集合および同一の`(Time, Step, Sequence)`集合を持たなければならない。
- `path.stem`を`filename`として使用するため、入力ファイル間で`path.stem`は一意でなければならない。
- 入力パスが空、ファイルが存在しない、または上記の要件を満たさない場合は例外を送出する。
  - 存在しないファイルには`FileNotFoundError`を使用する。
  - 値、スキーマおよび引数の不正には`ValueError`を使用する。

### 前処理仕様

- smoothingおよび規格化は入力検証後、間引きおよびflatten前に任意で適用する。
- 複数の前処理を併用する場合は、t方向smoothing、w方向smoothing、t方向規格化、w方向規格化の順に適用する。
- smoothingおよび規格化はスペクトル強度だけを変更し、`Time`、`Step`および`Sequence`の値、行数、波長列数は変更しない。

#### t方向smoothing

- `t_smoothing_window=None`の場合は適用しない。
- `t_smoothing_window`には有限かつ0より大きい値を指定する。それ以外は`ValueError`を送出する。
- 各`(Step, Sequence, w)`について、各行のTime値`t`を中心とした閉区間`[t - t_smoothing_window, t + t_smoothing_window]`に含まれるスペクトル強度の算術平均で、対象行のスペクトル強度を置き換える。
- 窓内の判定にはTimeの実値を使用するため、Timeが等間隔であることを要求しない。
- 区間端では存在する観測値だけを平均し、paddingおよび補間は行わない。
- 対象行自身を常に窓へ含める。

#### w方向smoothing

- `w_smoothing_window=None`の場合は適用しない。
- `w_smoothing_window`には有限かつ0より大きい値を指定する。それ以外は`ValueError`を送出する。
- 各`(Time, Step, Sequence)`について、各波長`w`を中心とした閉区間`[w - w_smoothing_window, w + w_smoothing_window]`に含まれるスペクトル強度の算術平均で、対象波長のスペクトル強度を置き換える。
- 窓内の判定には列名から得た波長の実値を使用するため、波長が等間隔であることを要求しない。
- 波長域の端では存在する波長だけを平均し、paddingおよび補間は行わない。
- 対象波長自身を常に窓へ含める。

#### t方向規格化

- `t_normalization_range=None`の場合は適用しない。
- t方向の規格化では、各`(Step, Sequence, w)`について、`t1`以上`t2`以下のTime区間に含まれるスペクトル強度の算術平均を求め、同じ`(Step, Sequence, w)`の全Timeにおけるスペクトル強度をその値で割る。
  - `t1`および`t2`は入力の`Time`列の値と同じ単位で指定し、`t1 <= t2`でなければならない。
  - `t1`および`t2`は有限値でなければならない。
  - 指定区間に該当するTimeがない場合、または算出した統計値が0もしくは有限値でない場合は`ValueError`を送出する。

#### w方向規格化

- `w_normalization_range=None`の場合は適用しない。
- w方向の規格化では、各`(Time, Step, Sequence)`について、`w1`以上`w2`以下の波長区間に含まれるスペクトル強度の算術平均を求め、同じ`(Time, Step, Sequence)`の全波長におけるスペクトル強度をその値で割る。
  - `w1`および`w2`は波長列名として解釈される数値と同じ単位で指定し、`w1 <= w2`でなければならない。
  - `w1`および`w2`は有限値でなければならない。
  - 指定区間に該当する波長がない場合、または算出した統計値が0もしくは有限値でない場合は`ValueError`を送出する。

### 間引き仕様

- 間引きはすべてのsmoothingおよび規格化の後、flattenの直前に適用する。
- t方向の間引き、w方向の間引きの順に適用する。
- Unique配列は公開APIの引数にせず、検証済みの全入力から内部で一度だけ生成し、すべての入力ファイルに共通して使用する。
- `t_downsampling_stride`および`w_downsampling_stride`はboolではない1以上の整数でなければならない。それ以外は`ValueError`を送出する。
- 間引き間隔が`1`の場合は対象をすべて残す。

間引き処理を担う内部関数は、対象のUnique配列と間引き間隔を引数として受け取る。

```python
def apply_t_downsampling(
    frame: pl.LazyFrame,
    unique_times: list[float],
    stride: int,
) -> pl.LazyFrame:
    ...


def apply_w_downsampling(
    frame: pl.LazyFrame,
    unique_wavelengths: list[float],
    stride: int,
) -> pl.LazyFrame:
    ...
```

#### t方向の間引き

- 全入力ファイルの`Time`列の値を結合し、重複を除いて数値として昇順に並べた配列をt方向のUnique配列とする。
- t方向のUnique配列に対して、先頭をindex 0として`0, t_downsampling_stride, 2 * t_downsampling_stride, ...`のindexにある値を残す。
- 選択した各`Time`値に一致する行を、`Step`および`Sequence`の値にかかわらずすべて残し、それ以外の行を削除する。
- 配列の末尾が選択indexに該当しない場合、末尾を追加で残さない。

#### w方向の間引き

- 全入力ファイルに共通する波長列名を数値の波長へ変換し、重複を除いて昇順に並べた配列をw方向のUnique配列とする。
- w方向のUnique配列に対して、先頭をindex 0として`0, w_downsampling_stride, 2 * w_downsampling_stride, ...`のindexにある値を残す。
- 選択した波長値に対応する波長列を残し、それ以外の波長列を削除する。`Time`、`Step`および`Sequence`列は削除しない。
- 配列の末尾が選択indexに該当しない場合、末尾を追加で残さない。

### flatten仕様

- 各入力パスは`Path(path).resolve()`で絶対パスへ正規化する。
- 入力ファイルの処理順は、正規化した絶対パスの文字列表現による昇順とする。
- flatten前に、波長、`Step`、`Sequence`、`Time`をそれぞれ数値として昇順に並べる。
- flatten後の特徴量列は、`(w, s, q, t)`をsort keyとして昇順に並べる。
- `Sequence`列の値を`q`とする。
- 特徴量列名は`f"{w}_{s}_{q}_{t}"`の形式とする。
- `w`は`f"{v:.1f}nm"`で表現される入力波長列名をそのまま使用する。
- `s`、`q` は int型で表現する。
- `t`は `f"{t:.2f}"` で表現する。
- 正規化後の特徴量列名が重複する場合は`ValueError`を送出する。
- flatten結果`df`の列順は、`filename`、flattenした特徴量列の順とする。
- `preprocess_and_flatten`が返すLazyFrameのスキーマは、collect後の`df`と同じとする。

### PCA仕様

- PCAの入力には`filename`を除くすべてのflatten特徴量列を使用する。
- `flatten_pca`は`sklearn.decomposition.PCA`をfitし、そのfit済みインスタンスを返す。
- `scaling_strategy="none"`相当とし、PCA前の中心化および尺度変換は行わない。
- 平均中心化は`sklearn.decomposition.PCA`内部の処理に任せ、分散による標準化は行わない。
- `n_component`には公開APIで受け取った値を指定し、追加の上限は設けない。
- 入力検証でnullとNaNを拒否するため、PCAのfitおよびtransform時に欠損値補完や行削除は行わない。外れ値処理も行わない。
- PCA solverはscikit-learnのデフォルトである`svd_solver="auto"`を使用する。
- `append_pca_scores`は、flatten特徴量を`pca.transform`へ渡して得たスコアを、入力の行順を維持して追加する。`pca.n_features_in_`と特徴量列数が異なる場合は`ValueError`を送出する。
- PCA成分の符号は一意に定まらないため、テストでは主成分やスコアの符号そのものを固定値と単純比較しない。

### PCA成分のreshape仕様

- `reshape_pca_components`は、flatten特徴量列の順序に対応する`pca.components_`を、component軸を先頭にして`(n_component, n_wavelengths, n_steps, n_sequences, n_times)`へreshapeした`np.ndarray`を返す。
- 各軸の値と順序はflatten仕様と同じく、波長、`Step`、`Sequence`、`Time`を数値昇順にしたものとする。
- reshape後の`result[c, wi, si, qi, ti]`は、component `c`（0始まり）の、波長`wi`、Step `si`、Sequence `qi`、Time `ti`に対応する係数とする。
- flatten特徴量がこれら4軸の直積を成さない場合、特徴量列名から座標を一意に復元できない場合、または`pca.n_features_in_`と特徴量列数が一致しない場合は`ValueError`を送出する。

### 出力要件

- `preprocess_and_flatten`をcollectした結果の行数は入力ファイル数nと一致し、各行は`filename`によって元の入力ファイルを一意に識別できなければならない。
- `append_pca_scores`をcollectした結果の列順は、`filename`、flatten特徴量列、`pca-1`から始まるPCAスコア列の順とする。
- 入力パスの指定順に依存せず、同じファイル集合から同じ行順および列順を得られなければならない。

### 完了条件

- 公開APIと公開される関数・クラスには型ヒントとNumPy形式のdocstringがある。
- 正常系、入力検証、`Sequence`、t/w方向smoothing、t/w方向規格化、t/w方向の間引き、並び順、LazyFrameの遅延実行、flatten結果、PCAのfit、PCAスコアの結合およびPCA成分のreshapeを対象とした自動テストがある。
- t/w方向の間引きテストは、間引き間隔`1`、`2`、入力不正、および末尾が選択indexに該当しない場合を対象とする。
- `tests/fixtures/real_subset`のParquetを使用し、複数ファイルを入力するend-to-endテストがある。
- PCAのテストは、符号反転を許容しながら分散、部分空間、再構成結果またはreshape後の係数配置を検証する。
- `uv run -m pytest`が成功する。
- `TASKS.md`に定義されたすべてのタスクが、対応する要件とテストを満たして完了している。
