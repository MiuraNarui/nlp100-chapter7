import csv
from collections import Counter

def load_data(path):
    data = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            text = row["sentence"]
            label = row["label"]

            # スペース区切りで単語に分割し、出現回数を数える
            feature = dict(Counter(text.split()))

            # 1事例を辞書オブジェクトとしてまとめる
            example = {
                "text": text,
                "label": label,
                "feature": feature
            }

            data.append(example)

    return data


# 学習データと検証データを辞書オブジェクトのリストに変換
train_data = load_data("SST-2/train.tsv")
dev_data = load_data("SST-2/dev.tsv")

# 学習データの最初の事例を確認
print(train_data[0])