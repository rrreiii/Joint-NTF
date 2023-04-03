# 大規模時系列文書における共通・非共通トピックとその時系列推移の同時抽出手法の提案
- 大規模な時系列情報付きの文書集合から共通・非共通トピックとそのトレンドの時間推移を同時に抽出するアプリです．　　
- Joint Nonnegative Tensor Factrization(Joint-NTF)というアルゴリズムを提案しています．　　
- アルゴリズムの詳細説明は"論文・発表資料"フォルダ内の資料を参照してください．
- 共同研究に用いたプログラムから公開可能な部分を切り出して整理したものをアップロードしています．

# アプリの概要
1. 分析対象の大規模時系列文書集合を読み込み，トレンドを時系列比較したい二つのドメイン(特許の取得会社など)を指定します．
2. 文書からトピックの成分となる用語（技術用語など）を抽出して辞書を作成します．
3. それぞれのドメインの文書集合を用語-文書-時間の三軸からなる特殊な三階のテンソルで表現します．
4. 提案するJoint-NTFの手法を用いて，二つのドメインの共通トピックと非共通トピックおよびそれらのトレンドの時系列推移を表す行列を得ます．
5. 得られた行列をグラフに可視化し，グラフと詳細な抽出内容をエクセルに出力します．

# 例）特許文書集合を分析する場合の想定用途
自社と競合他社が出願した特許文書集合を比較して，各企業の特許に共通して出現する技術トピックや各企業の特許に固有の技術トピックを検出し，それらの技術トピックの時間推移によるトレンドの変化を分析します．これにより，自社が現在は注力していないが競合他社が力を入れ始めている技術領域などを分析し,投資すべき領域の検討材料とする,などの利用方法を想定しています．

# フォルダの中身について
## NTF_algorithm/
### func_definitions/
- 　Joint-NTFの分析に必要なクラスや関数を定義したファイルを格納しているフォルダです．
### jupyter_workspace/
- 　個別の分析のためのipynb形式のファイルを格納しているフォルダです.func_definitions内のクラスや関数を呼び出しながら分析します．
### input_files/
- 　分析に用いるデータファイルの読み込み元のフォルダです．
### output_files/
- 　分析結果のエクセルファイルの出力先のフォルダです．
## 論文・発表資料/
- 本提案の学会論文と発表スライドのpdfを格納しているフォルダです．

# 今後のtodo
- 公開用ファイルの追加
  - スパーステンソルを用いた最適化プログラムを公開用に修正しアップロード
  - 生成データを用いたテスト分析を公開用に修正しアップロード
  - 時系列推移の将来予測結果を公開用に修正しアップロード
- webアプリ化
  - 環境構築や分析環境に不慣れな方でも利用できるようにする
  - インタラクティブにパラメータを調整しながら分析結果を確認できるようにする
- アルゴリズムの改善
  - pseudo-deflation methodの導入による精度の向上
  - 共通成分の抽出精度の改善方法の検討