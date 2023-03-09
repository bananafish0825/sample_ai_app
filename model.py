# 以下を「model.py」に書き込み
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# 10クラス分類（英語、日本語でそれぞれリストを作成しておく）
classes_ja = ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]
classes_en = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
n_class = len(classes_ja) # クラスの数を変数にまとめておく
img_size = 32 # 前sectionで扱ったCIFER-10に合わせて、32×32にする

# CNNのモデル（前sectionのモデルと同じもの）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def predict(img): # PILのimg(画像ファイル)を受け取る 関数を定義
    # モデルへの入力
    img = img.convert("RGB") # RGBに変換 CIFER-10で扱うのはRGBのカラー画像
    img = img.resize((img_size, img_size)) # リサイズ
    transform = transforms.Compose([transforms.ToTensor(), # tensorに変換
                                    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # 標準化:平均値を0、標準偏差を1に
                                ])
    img = transform(img)
    x = img.reshape(1, 3, img_size, img_size) # reshape:pytorchに入力可能な形に変換 1:batch size(1度に1枚の画像に処理を行う)、3:チャンネル数(RGB:赤緑青の3チャンネル)、高さ=img_size、幅=img_size

    # 訓練済みモデル
    net = Net()
    net.load_state_dict(torch.load( # モデルの読み込み
        "model_cnn.pth", map_location=torch.device("cpu") # 保存してあるモデルを予めアップロードしておく。streamlitクラウド上ではCPU上で動かすのでCPUに設定しておく必要がある
        ))
    
    # 予測
    net.eval()
    y = net(x) # 入力xを渡して出力yを得る

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # softmax関数で出力を確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)] # 日本語と英語でクラス名と確率を順番に並べたものを返す
