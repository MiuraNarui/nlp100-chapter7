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
            label = int(row["label"])
            feature = dict(Counter(text.split()))

            data.append({
                "text": text,
                "label": label,
                "feature": feature
            })

    return data


# 学習データを読み込む
train_data = load_data("SST-2/train.tsv")

# 特徴量とラベルを取り出す
train_features = [example["feature"] for example in train_data]
train_labels = [example["label"] for example in train_data]

# 辞書形式の特徴量を数値ベクトルに変換
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train_features)

# ロジスティック回帰モデルを学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# 特徴量名と重みを取得
feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]

# 特徴量名と重みを組にして、重みの小さい順に並べる
feature_weights = list(zip(feature_names, weights))
feature_weights.sort(key=lambda x: x[1])

# 重みの低い特徴量トップ20
print("重みの低い特徴量トップ20")
for feature, weight in feature_weights[:20]:
    print(feature, weight)

# 重みの高い特徴量トップ20
print("\n重みの高い特徴量トップ20")
for feature, weight in feature_weights[-20:][::-1]:
    print(feature, weight)