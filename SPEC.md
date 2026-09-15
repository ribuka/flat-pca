
## v1仕様

### スコープ

- v1では、複数のParquetファイルをflattenし、同じデータ集合にPCAをfitして得た主成分スコア列を追加した`pl.DataFrame`を返す。
- v1の公開APIは、学習済みPCAモデルの保存や未知データへのtransformを扱わない。
- t方向およびw方向のsmoothingと規格化は、任意の前処理としてv1の対象に含める。
- CLIおよびNotebookは成果物に含めない。

### 公開API

公開関数は次のシグネチャとする。

```python
def flatten_pca(
    paths: Sequence[str | Path],
    *,
    n_component: int | None = None,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    ...
```

- `paths`は1個以上のParquetファイルへのパスとする。
- `n_component`は1以上かつ`min(n_samples, n_features)`以下の整数、または`None`とする。`None`の場合はこの上限を使用する。
- `t_smoothing_window`はt方向smoothingの片側窓幅をTimeと同じ単位で指定し、`None`の場合は適用しない。
- `w_smoothing_window`はw方向smoothingの片側窓幅を波長と同じ単位で指定し、`None`の場合は適用しない。
- `t_normalization_range`はt方向規格化に用いる閉区間`(t1, t2)`を指定し、`None`の場合は適用しない。
- `w_normalization_range`はw方向規格化に用いる閉区間`(w1, w2)`を指定し、`None`の場合は適用しない。
- 戻り値はflatten結果にPCAスコア列を追加した`pl.DataFrame`とする。

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

- 前処理は入力検証後、flatten前に任意で適用する。
- 複数の前処理を併用する場合は、t方向smoothing、w方向smoothing、t方向規格化、w方向規格化の順に適用する。
- 前処理はスペクトル強度だけを変更し、`Time`、`Step`および`Sequence`の値、行数、波長列数は変更しない。

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

### flatten仕様

- 各入力パスは`Path(path).resolve()`で絶対パスへ正規化する。
- 入力ファイルの処理順は、正規化した絶対パスの文字列表現による昇順とする。
- flatten前に、波長、`Step`、`Sequence`、`Time`をそれぞれ数値として昇順に並べる。
- flatten後の特徴量列は、`(w, s, q, t)`をsort keyとして昇順に並べる。
- `Sequence`列の値を`q`とする。
- 特徴量列名は`f"{w}*{s}*{q}_{t}"`の形式とする。
- `w`は`f"{v:.1f}nm"`で表現される入力波長列名をそのまま使用する。
- `s`、`q` は int型で表現する。
- `t`は `f"{t:.2f}nm"` で表現する。
- 正規化後の特徴量列名が重複する場合は`ValueError`を送出する。
- flatten結果`df`の列順は、`filename`、flattenした特徴量列の順とする。

### PCA仕様

- PCAの入力には`filename`を除くすべてのflatten特徴量列を使用する。
- `src/spca/feature_engineering/pca.py`の`fit_and_transform_pca`を流用する。
- `scaling_strategy="none"`を指定し、PCA前の中心化および尺度変換は行わない。
- 平均中心化は`sklearn.decomposition.PCA`内部の処理に任せ、分散による標準化は行わない。
- `n_component`には公開APIで受け取った値を指定し、`max_n_component=None`として追加の上限を設けない。
- 入力検証でnullとNaNを拒否するため、既存コードには`impute_strategy="drop"`を指定する。この処理による行削除は発生しない前提とする。
- 外れ値処理は行わず、`outlier_strategy=None`を指定する。
- PCA solverは既存コードとscikit-learnのデフォルトである`svd_solver="auto"`を使用する。
- PCAスコアの列名は既存コードに合わせ、`pca-1`、`pca-2`、...、`pca-{n_component}`とする。
- `fit_and_transform_pca`が返す`pl.LazyFrame`をcollectし、公開APIから`pl.DataFrame`として返す。
- `fit_and_transform_pca`は入力列を保持したままPCAスコア列を追加するため、flatten結果との追加joinは行わない。
- PCA成分の符号は一意に定まらないため、テストでは主成分やスコアの符号そのものを固定値と単純比較しない。

### 出力要件

- 戻り値の行数は入力ファイル数nと一致しなければならない。
- 戻り値の各行は、`filename`によって元の入力ファイルを一意に識別できなければならない。
- 列順は、`filename`、flatten特徴量列、`pca-1`から始まるPCAスコア列の順とする。
- 入力パスの指定順に依存せず、同じファイル集合から同じ行順および列順を得られなければならない。

### 完了条件

- 公開APIと公開される関数・クラスには型ヒントとNumPy形式のdocstringがある。
- 正常系、入力検証、`Sequence`、t/w方向smoothing、t/w方向規格化、並び順、flatten結果、PCA結果を対象とした自動テストがある。
- `tests/fixtures/real_subset`のParquetを使用し、複数ファイルを入力するend-to-endテストがある。
- PCAのテストは、符号反転を許容しながら分散、部分空間または再構成結果を検証する。
- `uv run -m pytest`が成功する。
- `TASKS.md`に定義されたすべてのタスクが、対応する要件とテストを満たして完了している。
