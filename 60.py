import os
import urllib.request
import zipfile
import pandas as pd

# SST-2データセットをダウンロード
url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
zip_path = "SST-2.zip"

if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)

# ZIPファイルを展開
if not os.path.exists("SST-2"):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

# train.tsv と dev.tsv を読み込む
train = pd.read_csv("SST-2/train.tsv", sep="\t")
dev = pd.read_csv("SST-2/dev.tsv", sep="\t")

# ポジティブ(1)・ネガティブ(0)の件数を表示
print("train.tsv")
print(train["label"].value_counts().sort_index())

print("\ndev.tsv")
print(dev["label"].value_counts().sort_index())
