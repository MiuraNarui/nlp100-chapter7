import csv
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def load_data(path):
    data = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            text = row["sentence"]
            label = int(row["label"])
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

# 学習データの特徴量とラベル
train_features = [example["feature"] for example in train_data]
train_labels = [example["label"] for example in train_data]

# 検証データの特徴量とラベル
dev_features = [example["feature"] for example in dev_data]
dev_labels = [example["label"] for example in dev_data]

# 辞書形式の特徴量を数値ベクトルに変換
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train_features)
X_dev = vectorizer.transform(dev_features)

# ロジスティック回帰モデルを学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# 検証データを予測
predicted_labels = model.predict(X_dev)

# 混同行列を作成
cm = confusion_matrix(dev_labels, predicted_labels)

print("混同行列:")
print(cm)