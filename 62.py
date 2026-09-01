import csv
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


def load_data(path):
    data = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            text = row["sentence"]
            label = row["label"]

            # 61と同様に、単語の出現回数を特徴量にする
            feature = dict(Counter(text.split()))

            data.append({
                "text": text,
                "label": label,
                "feature": feature
            })

    return data


# 学習データを読み込む
train_data = load_data("SST-2/train.tsv")

# 特徴ベクトルとラベルを取り出す
features = [example["feature"] for example in train_data]
labels = [int(example["label"]) for example in train_data]

# 辞書形式の特徴量を、機械学習で使える数値ベクトルに変換
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(features)

# ロジスティック回帰モデルを学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, labels)

print("学習が完了しました")
print("学習事例数:", X_train.shape[0])
print("特徴量数:", X_train.shape[1])