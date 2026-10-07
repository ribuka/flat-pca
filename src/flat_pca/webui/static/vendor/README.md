# vendor

Web UI がネットワークなしで動くように、CDN を使わず同梱しているファイル。

| ファイル | 内容 | ライセンス |
| --- | --- | --- |
| `htmx.min.js` | [htmx](https://htmx.org/) | BSD Zero Clause License |
| `material-symbols-outlined.woff2` | [Material Symbols](https://fonts.google.com/icons)（Outlined）のうち、Web UI で使うアイコンだけを含むサブセット | Apache License 2.0（`LICENSE-material-symbols.txt`） |

`plotly.min.js` は Python の `plotly` パッケージに同梱されたものを `/static/vendor/plotly.min.js` で配信している（このディレクトリには置かない）。

## Material Symbols のアイコンを増やす

1. `scripts/fetch_material_symbols.py` の `ICON_NAMES` にアイコン名を追加する（名前は https://fonts.google.com/icons で調べる）。
2. リポジトリのルートで `uv run -m scripts.fetch_material_symbols` を実行し、`material-symbols-outlined.woff2` を取得し直す。
   - Google Fonts の CSS2 API に `&icon_names=...`（アルファベット順）を付けて、指定したアイコンだけのフォントを取得する。
   - 軸は `opsz`・`wght`・`FILL`・`GRAD` をすべて含めるので、`font-variation-settings` で塗りつぶしや太さを変えられる。
3. テンプレートでは `templates/macros/icon.html` の `icon` マクロで表示する。

```jinja
{% from "macros/icon.html" import icon %}
{{ icon("check_circle", class="icon-success") }}
```
