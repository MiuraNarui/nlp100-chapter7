import csv
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

# 特徴量とラベルを取り出す
train_features = [example["feature"] for example in train_data]
train_labels = [example["label"] for example in train_data]

dev_features = [example["feature"] for example in dev_data]
dev_labels = [example["label"] for example in dev_data]

# 辞書形式の特徴量を数値ベクトルに変換
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train_features)
X_dev = vectorizer.transform(dev_features)

# 正則化パラメータCを変化させる
# Cが小さいほど正則化が強く、Cが大きいほど正則化が弱い
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
accuracies = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, train_labels)

    # 検証データを予測
    dev_pred = model.predict(X_dev)

    # 正解率を計算
    accuracy = accuracy_score(dev_labels, dev_pred)
    accuracies.append(accuracy)

    print(f"C = {C}: 正解率 = {accuracy:.4f}")

# 正則化パラメータと正解率の関係をグラフ化
plt.figure(figsize=(8, 5))
plt.plot(C_values, accuracies, marker="o")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.title("Regularization Parameter C vs Validation Accuracy")
plt.grid(True)
plt.tight_layout()

# グラフを保存
plt.savefig("69.png", dpi=150)
plt.show()