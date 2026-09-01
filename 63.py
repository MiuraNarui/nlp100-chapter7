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
            feature = dict(Counter(text.split()))

            data.append({
                "text": text,
                "label": label,
                "feature": feature
            })

    return data


# 学習データと検証データを読み込む
train_data = load_data("SST-2/train.tsv")
dev_data = load_data("SST-2/dev.tsv")

# 学習データから特徴量とラベルを取り出す
train_features = [example["feature"] for example in train_data]
train_labels = [int(example["label"]) for example in train_data]

# 辞書形式の特徴量を数値ベクトルに変換
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train_features)

# ロジスティック回帰モデルを学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# 検証データの先頭事例を取得
first_dev = dev_data[0]

# 学習時と同じ変換器で特徴ベクトルに変換
X_dev_first = vectorizer.transform([first_dev["feature"]])

# ラベルを予測
predicted_label = model.predict(X_dev_first)[0]
true_label = int(first_dev["label"])

print("テキスト:", first_dev["text"])
print("予測ラベル:", predicted_label)
print("正解ラベル:", true_label)
print("一致しているか:", predicted_label == true_label)