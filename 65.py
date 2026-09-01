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


# 学習データを読み込む
train_data = load_data("SST-2/train.tsv")

# 学習データから特徴量とラベルを取り出す
train_features = [example["feature"] for example in train_data]
train_labels = [int(example["label"]) for example in train_data]

# 辞書形式の特徴量を数値ベクトルに変換
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train_features)

# ロジスティック回帰モデルを学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# 予測したいテキスト
text = "the worst movie I 've ever seen"

# テキストをBoWの特徴量に変換
feature = dict(Counter(text.split()))

# 学習時と同じ変換器で数値ベクトルに変換
X = vectorizer.transform([feature])

# ラベルを予測
predicted_label = model.predict(X)[0]

print("テキスト:", text)
print("予測ラベル:", predicted_label)

if predicted_label == 1:
    print("予測結果: ポジティブ")
else:
    print("予測結果: ネガティブ")