import csv
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def evaluate(true_labels, predicted_labels, name):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"{name}")
    print("正解率:", accuracy)
    print("適合率:", precision)
    print("再現率:", recall)
    print("F1スコア:", f1)
    print()


# 学習データと検証データを読み込む
train_data = load_data("SST-2/train.tsv")
dev_data = load_data("SST-2/dev.tsv")

# 特徴量とラベルを取り出す
train_features = [example["feature"] for example in train_data]
train_labels = [example["label"] for example in train_data]

dev_features = [example["feature"] for example in dev_data]
dev_labels = [example["label"] for example in dev_data]

# 辞書形式の特徴量を数値ベクトルに変換
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train_features)
X_dev = vectorizer.transform(dev_features)

# ロジスティック回帰モデルを学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# 学習データと検証データを予測
train_pred = model.predict(X_train)
dev_pred = model.predict(X_dev)

# 評価指標を計算
evaluate(train_labels, train_pred, "学習データ")
evaluate(dev_labels, dev_pred, "検証データ")