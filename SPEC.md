
## v1仕様

### スコープ

- v1では、複数のParquetファイルに前処理とflattenを適用し、その結果にPCAをfitする。
- `flatten_pca`はfit済みの`PcaModel`を返す。flatten結果およびPCAスコアを結合した`pl.DataFrame`は別の公開関数で取得する。
- v1の公開APIは、学習済みPCAモデルの永続化および未知データへのtransformを扱わない。
- t方向およびw方向のsmoothingと規格化は、任意の前処理としてv1の対象に含める。
- CLIおよびNotebookは成果物に含めない。

### 公開API

公開関数は次のシグネチャとする。`preprocess_and_flatten`は、従来の「前処理→flatten」部分を独立したAPIとして提供する。

```python
def preprocess_and_flatten(
    paths: Sequence[str | Path],
    *,
    target_steps: list[int] | None = None,
    edge_trim: list[float, float] | None = None,
    wavelength_range: tuple[float, float] | None = None,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
    t_downsampling_stride: int = 1,
    w_downsampling_stride: int = 1,
    materialize_once: bool = True,
) -> pl.LazyFrame:
    ...


def flatten_pca(
    paths: Sequence[str | Path] | None = None,
    *,
    flattened: pl.LazyFrame | None = None,
    n_component: int | None = None,
    target_steps: list[int] | None = None,
    edge_trim: list[float, float] | None = None,
    wavelength_range: tuple[float, float] | None = None,
    t_smoothing_window: float | None = None,
    w_smoothing_window: float | None = None,
    t_normalization_range: tuple[float, float] | None = None,
    w_normalization_range: tuple[float, float] | None = None,
    t_downsampling_stride: int = 1,
    w_downsampling_stride: int = 1,
    materialize_once: bool = True,
) -> PcaModel:
    ...


def append_pca_scores(
    pca_model: PcaModel,
    flattened: pl.LazyFrame,
) -> pl.LazyFrame:
    ...


def reshape_pca_components(
    pca_model: PcaModel,
    flattened: pl.LazyFrame,
) -> pl.DataFrame:
    ...
```

- `preprocess_and_flatten`の`paths`は1個以上のParquetファイルへのパスとする。
- `preprocess_and_flatten`は、`source`列およびflatten特徴量列を持つ`pl.LazyFrame`を返す。列順と特徴量列名は「flatten仕様」に従う。
- `target_steps`は対象とする`Step`列の値のリストを指定し、`None`の場合は`Step`列による行フィルタを適用しない。詳細は「Step行フィルタ仕様」に従う。
- `edge_trim`は`(edge_trim[0], edge_trim[1])`の2値を指定し、`None`の場合はStepTime・ReverseStepTime列を用いた行処理を適用しない。詳細は「StepTime列生成仕様」および「edge_trimによる行処理仕様」に従う。
- `wavelength_range`は`(wavelength_range[0], wavelength_range[1])`の閉区間を指定し、`None`の場合は波長列による列フィルタを適用しない。詳細は「wavelength_rangeによる列フィルタ仕様」に従う。
- `flatten_pca`が`paths`を指定する場合、`target_steps`、`edge_trim`および`wavelength_range`は`preprocess_and_flatten`と同じ意味・検証規則とする。`flattened`を指定する場合、`target_steps`、`edge_trim`および`wavelength_range`は使用しない。
- `flatten_pca`は`paths`または`flattened`のいずれか一方を受け取り、両引数の初期値は`None`とする。両方が`None`の場合、および両方が指定された場合は`ValueError`を送出する。
- `paths`を指定した場合、`paths`および前処理引数は`preprocess_and_flatten`と同じ意味・検証規則とする。内部で`preprocess_and_flatten`を呼び出す。
- `flattened`を指定した場合は、`source`列およびflatten特徴量列を持つ`pl.LazyFrame`を直接使用し、前処理およびflattenは行わない。
- `flatten_pca`は、選択したflatten結果の`source`以外の列を特徴量としてPCAをfitし、fit済みの`PcaModel`を返す。
- `n_component`は1以上かつ`min(n_samples, n_features)`以下の整数、または`None`とする。`None`の場合はこの上限を使用する。
- `t_smoothing_window`はt方向smoothingの片側窓幅をTimeと同じ単位で指定し、`None`の場合は適用しない。
- `w_smoothing_window`はw方向smoothingの片側窓幅を波長と同じ単位で指定し、`None`の場合は適用しない。
- `t_normalization_range`はt方向規格化に用いる閉区間`(t1, t2)`を指定し、`None`の場合は適用しない。
- `w_normalization_range`はw方向規格化に用いる閉区間`(w1, w2)`を指定し、`None`の場合は適用しない。
- `t_downsampling_stride`はt方向の間引き間隔を指定する。`1`の場合は間引かない。
- `w_downsampling_stride`はw方向の間引き間隔を指定する。`1`の場合は間引かない。
- `append_pca_scores`は、fit済み`pca_model`とflatten済み`flattened`を受け取り、`source`、flatten特徴量列およびPCAスコア列を持つ`pl.LazyFrame`を返す。スコア列は`pca-1`、`pca-2`、...、`pca-{n_component}`とする。`flattened`に残る欠損値（異なる入力ファイル間の`(Step, Sequence, StepTime)`集合の差異に由来するものを含む）は、fit時と同じ`pca_model.impute_strategy`に従って処理する（`transform_pca`と同じ規則）。`"drop"`の場合、欠損が残る行は出力から除外されるため、戻り値の行数が`flattened`の行数より少なくなることがある。`"median"`の場合、特徴量列の値は補完後の値になる。`"kmeans"`の場合、特徴量列の値はfit時に得たクラスタ中心を用いた補完後の値になる（詳細は「欠損値補完仕様」を参照）。欠損処理後に1行も残らない場合は`ValueError`を送出する。
- `materialize_once`（既定値`True`）が`True`の場合、`preprocess_and_flatten`はflatten完了後に一度だけ`.collect().lazy()`を行い、結果をメモリ上の`pl.DataFrame`起点の`pl.LazyFrame`として返す。これにより、戻り値をPCAのfitやスコア付与などで再利用しても、Parquet読み込みからflattenまでのクエリが再実行されない。`False`の場合は未実行のflattenクエリをそのまま返す。いずれの場合も戻り値の型は`pl.LazyFrame`であり、flatten結果・列順は一致する。`flatten_pca`が`paths`を指定する場合、`materialize_once`は同じ意味で内部の`preprocess_and_flatten`へ伝播する。`flattened`を指定する場合、`materialize_once`は使用しない。
- `reshape_pca_components`は、fit済み`pca_model`とflatten済み`flattened`を受け取り、後述の座標へ展開した`pca_model.pca.components_`を`pl.DataFrame`として返す。

### LazyFrameとメモリ使用

- `preprocess_and_flatten`では、Parquetの読み込みに`pl.scan_parquet`を使用し、前処理、間引きおよびflattenの列演算を可能な限り`pl.LazyFrame`のまま構成する。
- 入力検証、共通のUnique配列の確定、PCAのfit・transformなど、結果値が必要な境界でのみ必要最小限のcollectを行う。前処理済みの全入力フレームを一括で`pl.DataFrame`へmaterializeしてはならない。
- t/w方向規格化は、参照区間から算出した統計値の検証に実値を必要とするため、`pl.LazyFrame`を受け取った場合は入力ファイル単位で一度だけcollectし、結果を`.lazy()`で包み直して返す。これは`materialize_once`の値によらず適用される収集境界である。検証のためだけにcollectしたうえで未実行のクエリを返すと、呼び出し側のcollectで規格化より前段の処理がもう一度実行されるため、これを行ってはならない。
- `preprocess_and_flatten`、`append_pca_scores`は呼び出し側がcollectの時点と実行方法を選べるよう、常に`pl.LazyFrame`を返す。
- `preprocess_and_flatten`は既定（`materialize_once=True`）で、flatten完了後に一度だけcollectしメモリ上に固定した結果を`pl.LazyFrame`として返す。これはメモリ消費と引き換えに、戻り値を複数回collectしても前処理・flattenを再実行しないための挙動である。完全な遅延実行が必要な場合は`materialize_once=False`を指定する。
- `materialize_once=True`の場合、内部では入力ファイルごとに1回のPandas/Polars読み込みで値を取得するNumPyベースの高速経路（`numpy_preprocessing`）を用いる。まずメタデータ列（`Time`・`Step`・`Sequence`）のみを安価に先読みして、`target_steps`・StepTime生成・`edge_trim`適用後の共有Time配列およびwavelength_range適用後の共有波長配列を確定する。続いて各ファイルにつき`wavelength_range`で絞った波長列のみを1回読み込み、その読み込み結果に対して値検査（null・NaN・無限値）を行う（`validate_frame`相当の検査をこの読み込みに統合し、別読み込みでの重複検査は行わない）。smoothingおよび規格化は、間引き後に残る行・列と規格化の参照区間に必要な行・列の和集合に対してのみ計算し、各段のブール窓判定・グループ化・平均計算は`preprocess`パッケージの対応する関数（`apply_t_smoothing`等）と同一の演算をその位置に限定して行うため、結果は`materialize_once=False`側とビット単位で一致する。（(波長列数 × ファイル間でunionした`(Step, Sequence, StepTime)`組み合わせ数)に比例した個別のlazy式を積み上げる方式は、実用的なファイル数・波長数の組み合わせで著しく遅いため使用しない。）そのため`materialize_once=True`では、`preprocess_and_flatten`の呼び出し自体でflatten対象ファイルの読み込みと前処理・flatten処理が完了し、以降collectを遅延できるのはsparse列の間引き以降の段のみとなる。`materialize_once=False`の場合はこの限りではなく、`preprocess`パッケージの元のpolars実装をそのまま用い、flattenを含め呼び出し側の最初の`.collect()`までファイルの読み込みを一切行わない。ただし`t_normalization_range`または`w_normalization_range`を指定した場合は、前述の規格化の収集境界により、`materialize_once=False`であっても`preprocess_and_flatten`の呼び出し時点で各入力ファイルの読み込みと規格化までの前処理が実行される。flatten以降の段は`materialize_once=False`の規定どおり遅延する。

### 入力要件

- すべての入力ファイルに`Time`列、`Step`列および`Sequence`列が存在しなければならない。
- `Time`列、`Step`列および`Sequence`列は数値型でなければならない。
- `Time`列、`Step`列および`Sequence`列以外を波長列として扱う。
- 波長列名は、有限な数値`v`を`f"{v:.1f}nm"`で表現した形式でなければならない。
- スペクトル強度は数値型でなければならない。
- null、NaNおよび無限大を含む入力は受け付けない。ただし`wavelength_range`が指定されている場合、この判定は`Time`・`Step`・`Sequence`列と、指定区間内の波長列のみを対象とし、区間外の波長列の値はこの判定の対象としない（区間外の列は後段の`wavelength_range`による列フィルタで削除されるため）。必須列の存在・数値型・波長列名フォーマット・ファイル間の波長集合一致など、値以外の要件は`wavelength_range`の指定に関わらず全波長列を対象に判定する。
- 各ファイル内で`(Time, Step, Sequence)`の組み合わせは一意でなければならない。
- すべてのファイルは、同一の波長集合および同一の`(Time, Step, Sequence)`集合を持たなければならない。
- `source`列には正規化済み入力パスの`path.as_posix()`を使用する。正規化パスは`Path.resolve()`により一意であるため、`path.stem`の一意性は`source`列の一意性には必要としない。
- `load_and_validate_inputs`（および`preprocess_and_flatten`・`flatten_pca`）は`stem_uniqueness`引数（`"skip"` | `"warn"` | `"error"`、既定値`"skip"`）で入力ファイル間の`path.stem`重複チェックの有無を選択できる。`"skip"`は重複を許容してチェックを行わず、`"warn"`は重複時に警告を出しつつ処理を継続し、`"error"`は重複時に`ValueError`を送出する。
- 入力パスが空、ファイルが存在しない、または上記の要件を満たさない場合は例外を送出する。
  - 存在しないファイルには`FileNotFoundError`を使用する。
  - 値、スキーマおよび引数の不正には`ValueError`を使用する。

### Step行フィルタ仕様

- 入力検証は各ファイルのParquetを読み込んだ直後の元データに対して行い、「入力要件」の各項目は`target_steps`による行フィルタ適用前の状態を対象に判定する。
- Step行フィルタは、入力検証の直後、StepTime列生成より前に適用する。
- `target_steps`が`None`でない場合、各ファイルについて`pl.col("Step").is_in(target_steps)`を満たさない行を削除する。
- `target_steps`が`None`の場合は行フィルタを適用せず、全行を保持する。
- すべての入力ファイルに同一の`target_steps`を適用するため、フィルタ後も各ファイル間で行集合の対応関係は維持される。

### StepTime列生成仕様

- StepTime列は、Step行フィルタ適用後、`Time`列の直後に挿入する新しい数値列とする。
- 各ファイル内の行を`Time`昇順に並べたとき、直前行に対して`Step`または`Sequence`の値が変化した行を、新しい区間の先頭行とする。先頭行自身も区間の一部とする。
- 区間の先頭行の`StepTime`は`0.00`とする。
- 区間内の各行の`StepTime`は、その行の`Time`値から区間先頭行の`Time`値を引いた差分とする。すなわちStepTime列は、`Step`と`Sequence`の組が変化しない区間ごとに独立した経過時間を表す。
- ReverseStepTime列は、StepTime列と同じ区間分割を用い、区間内の`StepTime`値の並びをTime降順に対応させた（区間内で逆順に並べ替えた）値を持つ新しい数値列とする。区間の末尾行の`ReverseStepTime`は常に`0.00`となる。
- StepTimeおよびReverseStepTime列の生成は、t/w方向smoothing、t/w方向規格化、間引きおよびflattenより前に適用する。
- flatten処理における特徴量列名の`t`は、`Time`列の値ではなく`StepTime`列の値を使用する。「flatten仕様」を参照する。

### edge_trimによる行処理仕様

- `edge_trim`が`None`でない場合、StepTimeおよびReverseStepTime列を生成した直後に、以下の上書き処理を適用する。`None`の場合は適用しない。
- meta列は`Time`、`Step`、`StepTime`、`ReverseStepTime`、`Sequence`の5列を指す。
- `StepTime`が`edge_trim[0]`より小さい行について、meta列以外の列（波長列を含む）の値を`None`で上書きする。
- `ReverseStepTime`が`edge_trim[1]`より小さい行について、meta列以外の列（波長列を含む）の値を`None`で上書きする。
- 上記2条件は独立に判定し、両方に該当する行はどちらの上書きも適用された結果となる。
- この上書きは、t/w方向smoothing、t/w方向規格化、間引きおよびflattenより前に適用する。

### wavelength_rangeによる列フィルタ仕様

- `wavelength_range`が`None`でない場合、StepTime・ReverseStepTime列を生成した直後、`edge_trim`による行処理より前に、以下の列選択を適用する。`None`の場合は適用しない。
- `wavelength_range[0]`（下限）以上`wavelength_range[1]`（上限）以下の波長を持つ波長列だけを残し、それ以外の波長列を削除する。`Time`、`Step`、`Sequence`、`StepTime`、`ReverseStepTime`列は削除しない。
- `wavelength_range[0]`および`wavelength_range[1]`は有限値でなければならず、`wavelength_range[0] <= wavelength_range[1]`でなければならない。それ以外は`ValueError`を送出する。
- 指定区間に該当する波長列が1つも無い場合は`ValueError`を送出する。
- この列選択は、`edge_trim`による行処理、t/w方向smoothing、t/w方向規格化、間引き（Unique配列の生成を含む）およびflattenより前に適用するため、以降のすべての処理は列フィルタ後の波長列だけを対象とする。
- 「入力要件」のnull・NaN・無限大チェックは、この列選択の適用前（各ファイルのParquetを読み込んだ直後）に行うが、`wavelength_range`が指定されている場合はその判定範囲を`Time`・`Step`・`Sequence`列と区間内の波長列に限定する。区間外の波長列にnull・NaN・無限大が含まれていても、この列選択で削除される限りエラーにはならない。

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
- flatten前に、波長、`Step`、`Sequence`、`StepTime`をそれぞれ数値として昇順に並べる。
- flatten後の特徴量列は、`(w, s, q, t)`をsort keyとして昇順に並べる。
- `Sequence`列の値を`q`とする。
- 特徴量列名は`f"{w}_{s}_{q}_{t}"`の形式とする。
- `w`は`f"{v:.1f}nm"`で表現される入力波長列名をそのまま使用する。
- `s`、`q` は int型で表現する。
- `t`は`StepTime`列の値を`f"{t:.2f}"`で表現する。`Time`列の値は使用しない。
- 正規化後の特徴量列名が重複する場合は`ValueError`を送出する。
- flatten結果`df`の列順は、`source`、flattenした特徴量列の順とする。
- `preprocess_and_flatten`が返すLazyFrameのスキーマは、collect後の`df`と同じとする。

### PCA仕様

- PCAの入力には`source`を除くすべてのflatten特徴量列を使用する。
- `flatten_pca`は`fit_pca`でPCAをfitし、そのfit済み`PcaModel`を返す。
- `fit_flattened_pca`は、PCAのfitに`src/flat_pca/feature_engineering/pca.py`の`fit_pca`を使用しなければならない。`sklearn.decomposition.PCA`を直接生成・fitしてはならない。
- `fit_flattened_pca`は`fit_pca`に、flatten特徴量列、`impute_strategy`（既定値`"drop"`）、`impute_kmeans_n_clusters`（既定値`None`）、`outlier_strategy=None`、`scaling_strategy="none"`および`max_n_component=None`を指定し、返却された`PcaModel`をそのまま返す。`impute_strategy`および`impute_kmeans_n_clusters`は公開APIの`flatten_pca`から`fit_flattened_pca`を経てそのまま`fit_pca`へ転送される。
- `scaling_strategy="none"`相当とし、PCA前の中心化および尺度変換は行わない。
- 平均中心化は`sklearn.decomposition.PCA`内部の処理に任せ、分散による標準化は行わない。
- `n_component`には公開APIで受け取った値を指定し、追加の上限は設けない。
- 入力検証でnullとNaNを拒否するため、PCAのfitおよびtransform時に欠損値補完や行削除は行わない。外れ値処理も行わない。
- PCA solverはscikit-learnのデフォルトである`svd_solver="auto"`を使用する。
- `append_pca_scores`は、flatten特徴量を`pca_model.pca.transform`へ渡して得たスコアを、入力の行順を維持して追加する。`pca_model.pca.n_features_in_`と特徴量列数が異なる場合は`ValueError`を送出する。
- PCA成分の符号は一意に定まらないため、テストでは主成分やスコアの符号そのものを固定値と単純比較しない。

### 欠損値補完仕様

- `fit_pca`の`impute_strategy`は`"drop"`、`"median"`または`"kmeans"`のいずれかとする。それ以外は`ValueError`を送出する。
- `impute_kmeans_n_clusters`は`impute_strategy="kmeans"`のときのみ指定できる整数（またはこの場合に限り`None`）とする。`impute_strategy`が`"kmeans"`以外のときに`None`でない値を指定した場合、または`bool`を含む0以下の値を指定した場合は`ValueError`を送出する。
- `"drop"`および`"median"`の挙動は従来通りとする。`"drop"`は`columns`のいずれかにnullまたはNaNを含む行を除外する。`"median"`はfit時に算出した列ごとの中央値でnullおよびNaNを埋める。
- `"kmeans"`は次の手順で欠損値を補完する。
  1. `columns`にnullまたはNaNを1つも含まない行を「完全行」、1つ以上含む行を「欠損行」とする。
  2. fit時、完全行が0件の場合は`ValueError`を送出する。
  3. クラスタ数は、`impute_kmeans_n_clusters`（`None`の場合は既定値8）を、完全行数を超えないよう`min(値, 完全行数)`にクリップして決定する。
  4. 完全行の`columns`（スケーリング前の生値）に対し、決定的な`random_state`を固定した`sklearn.cluster.KMeans`をfitし、`cluster_centers_`をクラスタ中心として得る。
  5. 各欠損行について、欠損していない列のみを用いた二乗ユークリッド距離（部分距離）が最小のクラスタ中心を選び、そのクラスタ中心の値で欠損列を埋める。完全行の値は変更しない。
  6. 出力の行順および行数は入力と同じとする。
- `transform_pca`（`append_pca_scores`を含む）が`impute_strategy="kmeans"`の`pca_model`を用いて欠損値を補完する場合、fit時に得たクラスタ中心をそのまま再利用し、再fitは行わない。上記5の手順のみを適用する。
- `PcaModel`は`impute_kmeans_n_clusters`（fit時に確定したクラスタ数、`"kmeans"`以外では`None`）と`impute_kmeans_centroids`（クラスタごとの列→値マッピングのリスト、`"kmeans"`以外では空リスト）を保持する。`to_transform_payload`・`from_transform_payload`・`to_transform_json`・`from_transform_json`はこれらを含めて往復可能とする。

### PCA成分のreshape仕様

- `reshape_pca_components`は、flatten特徴量列の順序に対応する`pca_model.pca.components_`を、component軸、波長、`Step`、`Sequence`、`StepTime`の座標へ展開したlong形式の`pl.DataFrame`を返す。
- 返却するDataFrameの列順は`StepTime`、`Step`、`Sequence`、`wavelength`、`component`、`coefficient`とする。`component`は1始まりの整数、`wavelength`は数値、`coefficient`は対応するPCA係数とする。
- 行順は`Step`、`Sequence`、`StepTime`、`component`、`wavelength`を数値昇順にした順とする。
- 各軸の値と順序はflatten仕様と同じく、波長、`Step`、`Sequence`、`StepTime`を数値昇順にしたものとする。
- 返却DataFrameの各行は、component `c`（1始まり）の、波長`wavelength`、Step `Step`、Sequence `Sequence`、StepTime `StepTime`に対応する係数を`coefficient`へ保持する。
- flatten特徴量がこれら4軸の直積を成さない場合、特徴量列名から座標を一意に復元できない場合、または`pca_model.pca.n_features_in_`と特徴量列数が一致しない場合は`ValueError`を送出する。

### 出力要件

- `preprocess_and_flatten`をcollectした結果の行数は入力ファイル数nと一致し、各行は`source`によって元の入力ファイルを一意に識別できなければならない。
- `append_pca_scores`をcollectした結果の列順は、`source`、flatten特徴量列、`pca-1`から始まるPCAスコア列の順とする。
- 入力パスの指定順に依存せず、同じファイル集合から同じ行順および列順を得られなければならない。

### 完了条件

- 公開APIと公開される関数・クラスには型ヒントとNumPy形式のdocstringがある。
- 正常系、入力検証、`Sequence`、Step行フィルタ、StepTime・ReverseStepTime列生成、edge_trimによる行処理、wavelength_rangeによる列フィルタ、t/w方向smoothing、t/w方向規格化、t/w方向の間引き、並び順、LazyFrameの遅延実行、flatten結果、PCAのfit、PCAスコアの結合およびPCA成分のreshapeを対象とした自動テストがある。
- t/w方向の間引きテストは、間引き間隔`1`、`2`、入力不正、および末尾が選択indexに該当しない場合を対象とする。
- `tests/fixtures/real_subset`のParquetを使用し、複数ファイルを入力するend-to-endテストがある。
- PCAのテストは、符号反転を許容しながら分散、部分空間、再構成結果またはreshape後の係数配置を検証する。
- `uv run -m pytest`が成功する。
- `TASKS.md`に定義されたすべてのタスクが、対応する要件とテストを満たして完了している。
