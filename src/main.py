import flet as ft
import cv2
import numpy as np
import base64
import os
import tensorflow as tf
from tensorflow import keras

# CNNモデルの学習 (グローバルスコープで一度だけ行う簡易実装)
# 本来は学習済みモデルを保存・読み込みすべきだが、デモ用に起動時に毎回学習する。
def train_cnn_model():
    """
    MNISTデータセットを用いてCNNモデルを学習します。
    
    アプリの起動時に一度だけ呼び出され、手書き数字認識用のモデルを構築・学習して返します。
    デモ用のため、エポック数は少なく設定しています。

    Returns:
        keras.Model: 学習済みのCNNモデル
    """
    print("MNISTデータの読み込み中...")
    # MNISTデータのロード (訓練用とテスト用)
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # 正規化とリシェイプ
    # ピクセル値を 0-255 から 0.0-1.0 に変換
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    # Conv2D層への入力のために (サンプル数, 高さ, 幅, チャンネル数) の4次元配列に変形
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    print("CNNモデルの構築と学習中...")
    # モデルの定義 (Sequentialモデル)
    model = keras.Sequential([
        # 畳み込み層: 3x3のフィルタを32個使用、活性化関数はReLU
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # プーリング層: 2x2の領域で最大をとって圧縮 (ダウンサンプリング)
        keras.layers.MaxPooling2D((2, 2)),
        # 平坦化層: 1次元のベクトルに変換
        keras.layers.Flatten(),
        # 全結合層: 128ノード、活性化関数はReLU
        keras.layers.Dense(128, activation='relu'),
        # 出力層: 10クラス(0-9)への確率を出力 (Softmax)
        keras.layers.Dense(10, activation='softmax')
    ])

    # モデルのコンパイル
    # optimizer: adam (一般的によく使われる)
    # loss: sparse_categorical_crossentropy (整数の教師データに対する多値分類)
    # metrics: accuracy (正解率)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 学習の実行
    # 時間短縮のためepochs=3に設定 (実用的な精度には不十分だがデモとしては動作する)
    model.fit(X_train, y_train, epochs=3, verbose=1)
    print("学習完了")
    return model

# アプリ起動時に学習を実行 (少し時間がかかる)
cnn_model = train_cnn_model()

def main(page: ft.Page):
    """
    アプリケーションのメインエントリポイント。
    UIの構築とイベントハンドリングを設定します。

    Args:
        page (ft.Page): Fletのページオブジェクト
    """
    page.title = "Real Handwriting Recognition (CNN)"
    page.scroll = ft.ScrollMode.AUTO

    # 画像を表示するコントロール
    img_original = ft.Image(src="", width=300, height=300, fit=ft.ImageFit.CONTAIN, visible=False)
    img_processed = ft.Image(src_base64="", width=200, height=200, fit=ft.ImageFit.CONTAIN, visible=False)
    
    # 判定結果を表示するテキスト
    result_text = ft.Text("", size=30, weight="bold", color="green")
    status_text = ft.Text("画像をアップロードしてください")

    # 画像アップロード用のディレクトリを作成
    upload_dir = os.path.join(os.path.dirname(__file__), "assets", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    def process_image(file_path):
        """
        OpenCVを使用して画像の前処理を行います。
        手書き数字の写真をMNIST形式 (28x28, 黒背景白文字) に変換します。

        Args:
            file_path (str): 処理する画像のパス

        Returns:
            tuple: (final_img, img_str, msg)
                - final_img (numpy.ndarray): モデル入力用の画像データ (28x28)
                - img_str (str): 表示用のBase64エンコード文字列
                - msg (str): 処理結果のメッセージ
                ※ エラー時は (None, None, error_msg) を返す
        """
        try:
            # 1. 画像の読み込み
            # 日本語パス対応のため、np.fromfile で読み込んで imdecode する
            with open(file_path, "rb") as f:
                bytes = bytearray(f.read())
                numpy_array = np.asarray(bytes, dtype=np.uint8)
                img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        except Exception as e:
            return None, None, f"画像の読み込みエラー: {str(e)}"

        if img is None:
            return None, None, "画像の読み込みに失敗しました"

        # 2. グレースケール変換
        # カラー情報を捨てて、明暗だけの画像にする
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. ノイズ除去
        # ガウシアンフィルタで少しぼかすことで、紙のザラつきなどの高周波ノイズを消す
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. 2値化 (閾値処理) & 反転
        # 白黒の2色にはっきりと分ける。
        # THRESH_BINARY_INV: 白い紙に黒い文字 → 黒い背景に白い文字 (MNISTの形式に合わせるため反転)
        # THRESH_OTSU: 最適な閾値を自動で計算するアルゴリズム
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 5. 輪郭抽出
        # 白い塊（文字と思われる部分）の外形を見つける
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, "文字が見つかりませんでした"

        # 最も面積が大きい輪郭を「文字」とみなして取得
        c = max(contours, key=cv2.contourArea)
        # 輪郭を囲む長方形 (x, y, 幅, 高さ) を取得
        x, y, w, h = cv2.boundingRect(c)

        # 6. 文字領域の切り出し (ROI: Region of Interest)
        # 少し余裕 (padding) を持たせて切り取る
        pad = 20
        h_img, w_img = binary.shape
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + pad * 2)
        h = min(h_img - y, h + pad * 2)

        roi = binary[y:y+h, x:x+w]
        
        # 7. アスペクト比を維持してリサイズ
        # 28x28の枠の中に、最大20x20程度になるように縮小する (MNISTデータ作成時の慣習)
        target_size = 20
        h_roi, w_roi = roi.shape
        scale = target_size / max(h_roi, w_roi)
        new_w, new_h = int(w_roi * scale), int(h_roi * scale)
        resized_roi = cv2.resize(roi, (new_w, new_h))

        # 8. センタリング配置
        # 28x28の真っ黒な画像を作成
        final_img = np.zeros((28, 28), dtype=np.uint8)
        # その中心にリサイズした文字画像を配置
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        final_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized_roi

        # 9. 表示用にエンコード
        # NumPy配列をPNG画像データに変換し、さらにBase64文字列にする（HTMLimgタグ用）
        _, buffer = cv2.imencode('.png', final_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # 予測用にfinal_img (numpy array) も返す
        return final_img, img_str, "処理完了"

    def on_file_picked(e: ft.FilePickerResultEvent):
        """
        ファイル選択ダイアログが閉じられたときに呼ばれるイベントハンドラ。
        選択されたファイルをサーバーにアップロードします。
        """
        try:
            if e.files and len(e.files) > 0:
                print(f"File picked: {e.files[0].name}")
                
                # UIを先に更新
                status_text.value = "アップロード中..."
                result_text.value = "" # 前回の結果をクリア
                status_text.update()
                result_text.update()
                
                # その後でアップロード開始
                file_picker.upload(e.files)
        except Exception as ex:
            print(f"Error in on_file_picked: {ex}")
            status_text.value = f"エラーが発生しました: {ex}"
            status_text.update()

    def on_upload_complete(e: ft.FilePickerUploadEvent):
        """
        ファイルのアップロードが完了したときに呼ばれるイベントハンドラ。
        画像処理を実行し、AIで予測を行い、結果を表示します。
        """
        try:
            print(f"Upload complete for: {e.file_name}")
            file_name = e.file_name
            file_path = os.path.join(upload_dir, file_name)
            
            status_text.value = "処理・判定中..."
            status_text.update()

            # 画像処理の実行
            processed_data = process_image(file_path)
            
            # エラーチェック
            if processed_data[0] is None:
                 error_msg = processed_data[2]
                 status_text.value = error_msg
                 img_processed.visible = False
                 page.update()
                 return

            final_img_array, result_base64, msg = processed_data
            
            status_text.value = msg
            
            # Base64画像を表示 (元画像)
            with open(file_path, "rb") as f:
                original_base64 = base64.b64encode(f.read()).decode('utf-8')
            img_original.src_base64 = original_base64
            img_original.src = ""
            img_original.visible = True
            
            # 処理後画像を表示
            img_processed.src_base64 = result_base64
            img_processed.visible = True

            # --- AIによる判定 (予測) ---
            # モデル入力用に正規化 (0.0-1.0)
            input_data = final_img_array.astype("float32") / 255.0
            # (バッチサイズ, 高さ, 幅, チャンネル) の4次元に変形
            input_data = input_data.reshape(1, 28, 28, 1)
            
            # 予測実行
            try:
                prediction = cnn_model.predict(input_data)
                # 最も確率が高いインデックス (=予測数字) を取得
                predicted_digit = prediction.argmax()
                # その確率を取得
                probability = prediction[0][predicted_digit]
                
                # 結果表示
                result_text.value = f"AIの判定: {predicted_digit} (確信度: {probability:.1%})"
                status_text.value = "判定完了！"
            except Exception as e_predict:
                print(f"Prediction error: {e_predict}")
                status_text.value = f"AI判定エラー: {e_predict}"
            
            page.update()
            
            # 後始末: アップロードされたファイルを削除
            # (デモ用なのでサーバーのディスクを圧迫しないように、かつ上書きトラブル回避のため)
            try:
                os.remove(file_path)
                print(f"Removed temporary file: {file_path}")
            except Exception as e_rm:
                print(f"Failed to remove file: {e_rm}")

        except Exception as ex:
            print(f"Error in on_upload_complete: {ex}")
            status_text.value = f"処理エラー: {ex}"
            page.update()

    # FilePickerコントロールの初期化
    file_picker = ft.FilePicker(on_result=on_file_picked, on_upload=on_upload_complete)
    # オーバーレイに追加しないとダイアログが表示されない
    page.overlay.append(file_picker)

    # UIレイアウトの定義
    page.add(
        ft.SafeArea(
            ft.Column(
                [
                    ft.Text("手書き文字認識アプリ (CNN)", size=24, weight="bold"),
                    ft.ElevatedButton(
                        "画像をアップロード",
                        icon=ft.Icons.UPLOAD_FILE,
                        on_click=lambda _: file_picker.pick_files(
                            allow_multiple=False,
                            file_type=ft.FilePickerFileType.IMAGE
                        )
                    ),
                    status_text,
                    result_text,
                    ft.Divider(),
                    ft.Row(
                        [
                            ft.Column([
                                ft.Text("元の写真"),
                                ft.Container(content=img_original, border=ft.border.all(1, "grey"), border_radius=10, padding=10),
                            ], horizontal_alignment="center"),
                            
                            ft.Icon(ft.Icons.ARROW_FORWARD, size=40),
                            
                            ft.Column([
                                ft.Text("AI入力用 (28x28)"),
                                ft.Container(content=img_processed, border=ft.border.all(1, "red"), bgcolor="black", padding=10),
                            ], horizontal_alignment="center"),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
        )
    )

if __name__ == "__main__":
    # upload_dir を絶対パスで指定してアプリを起動
    import os
    abs_upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "uploads"))
    print(f"Upload directory: {abs_upload_dir}")
    ft.app(target=main, upload_dir=abs_upload_dir)
