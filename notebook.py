# %% [markdown]
# # Red Wine Quality Classification

# %% [markdown]
# ## Import and Load Dataset
# We need to import some library that will be used in data preprocessing, data manipulation, and modeling.

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# Load the red wine dataset. The dataset is from UCI Machine Learning and can be downloaded at Kaggle: 
# 
# https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

# %%
df = pd.read_csv("dataset/winequality-red.csv")
df.head()

# %% [markdown]
# ## Data Understanding

# %% [markdown]
# First we want to find basic information from the dataset, like shape, data type, general stats, null values, and duplicate data. Than we will doing some EDA.

# %%
df.shape

# %%
df.info()

# %%
df.describe()

# %%
df.isna().sum()

# %%
df.duplicated().sum()

# %% [markdown]
# Next, we would like to see the distribution of each feature in the dataset. This will be usefull to determine which feature is distributed inbalance.

# %%
for i in df.columns:
    df[i].hist(bins=10, figsize=(8,6))
    plt.title(f"Distribution of {i}")
    plt.savefig(f"docs/distribution_{i}.png")
    plt.show()

# %% [markdown]
# From the distribution above, some feature like density and pH seems like have shape of normal distribution. But the other features, like alcohol, residual sugar have right-skewed and feature like chlorides have left-skewed type.  

# %% [markdown]
# Next, we will try to some the effect or distribution of each feature vs the quality of the wine.

# %%
for idx, i in enumerate(df.columns):
    if i == "quality":
        continue
    plt.figure(figsize=(8, 6))
    plt.title(f"Feature {idx+1}: {i} vs quality")
    sns.barplot(x="quality", y=i, data=df, hue="quality", legend=False)
    plt.savefig(f"docs/feature_{i}_vs_quality.png")

# %% [markdown]
# From the some of visualization above, we can see that some features like alcohol, sulphates, citric acid, and volatile acidity are highly affecting the quality of the wine (either better of worst).

# %% [markdown]
# Next, we will try to understand the distribution of each feature, so we can analyze the outliers and remove them.

# %%
for i in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=i, data=df)
    plt.savefig(f"docs/boxplot_{i}.png")

# %% [markdown]
# From the boxplot above, we can see that some of important features like alcohol or sulphates still contain outliers. So we need remove them in later on data processing. Next we will confirm the visualization above using heatmap.

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True)
plt.savefig("docs/heatmap.png")

# %% [markdown]
# From the heatmap above, we found the same thing, that feature like alcohol and sulphates have high correlation to quality of the wine.

# %% [markdown]
# ## Data Preparation
# After doing some analysis, we will do data preparation like removing null values, duplicates, and outliers. We also will normalize the feature to make them have equall weights for the ML models.

# %%
clean_df = df.copy()

# %%
clean_df.isna().sum()

# %% [markdown]
# The dataset is already clean, so we don't need to drop any null values. But we will still need to clean the duplicates data and outlier especially from important features. Duplicates data sometimes make the data inbalance.

# %%
clean_df = clean_df.drop_duplicates()
clean_df.duplicated().sum()

# %% [markdown]
# After cleaning duplicate values, we will removing outlier from some of the important features. This is because outliers can make bias for the model.

# %%
corr = clean_df.corr()["quality"].sort_values(ascending=False)
corr

# %%
# select that have correlation > 0.2 (either positive or negative, but not including target values)
selected_cols = corr[abs(corr) > 0.2].index.tolist()
selected_cols.remove("quality")
selected_cols

# %%
# drop outliers of that features using IQR
for i in selected_cols:
    Q1 = clean_df[i].quantile(0.25)
    Q3 = clean_df[i].quantile(0.75)
    IQR = Q3 - Q1
    clean_df = clean_df[~((clean_df[i] < (Q1 - 1.5 * IQR)) | (clean_df[i] > (Q3 + 1.5 * IQR)))]

# %% [markdown]
# Next, we also need to convert the target values from 10 categories into 3 categories, because we want to classify whether the wine is having a bad, medium/average, or a good quality. This is also make the result more easy to deliver to the customer.

# %%
clean_df["quality"] = pd.cut(clean_df["quality"], bins=[0, 4, 6, 10], labels=["bad", "medium", "good"])
clean_df

# %%
clean_df["quality"].value_counts()

# %%
# encode the target values, use 0 for bad, 1 for medium, and 2 for good
clean_df["quality"] = clean_df["quality"].map({"bad": 0, "medium": 1, "good": 2})

# %%
clean_df.head(3)

# %% [markdown]
# The features are already cleaned, now we need to split for training and testing. Then we need transform it using standard scaler to make the data have equal weight, so it can increase the performance and reduce the bias of the ML models.

# %%
X = clean_df.drop("quality", axis=1)
y = clean_df["quality"]

# split data with 20% test size
# also use random_state to make it reproduceable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=14)

# %%
# fit transform X_train, but not X_test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ## Modeling
# Because the data processing is already done, next we will train a model to do the classification. We will use 3 models, KNN, SVM, and Random Forest. Those model are choosen because it some of the most common model used for classification.

# %%
# helper list to store the result
y_pred_list = []

# %%
# create knn model for classification
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# %%
y_pred_list.append(y_pred)

# %%
# display knn model performance
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred, zero_division=0)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

# %%
# now create model using svm (svc for classification)
svc = SVC(C=1.0, kernel="rbf", gamma="scale")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

# %%
y_pred_list.append(y_pred)

# %%
# display svm model performance
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred, zero_division=0)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")


# %%
# last, model using random forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# %%
y_pred_list.append(y_pred)

# %%
# display rf model performance
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred, zero_division=0)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

# %% [markdown]
# ## Evaluation

# %% [markdown]
# For evaluation, we will compare all of those model and use the accuracy, precision, recall, and F1 metrics to evaluate the model.

# %%
models = ['KNN', 'SVM', 'Random Forest']

accuracy = [accuracy_score(y_test, y_pred) for y_pred in y_pred_list]
precision = [precision_score(y_test, y_pred, average='weighted', zero_division=0) for y_pred in y_pred_list]
recall = [recall_score(y_test, y_pred, average='weighted', zero_division=0) for y_pred in y_pred_list]
f1 = [f1_score(y_test, y_pred, average='weighted') for y_pred in y_pred_list]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}

for i, (metric_name, metric_values) in enumerate(metrics.items()):
    ax = axs[i // 2, i % 2]
    bars = ax.bar(models, metric_values, color=['blue', 'green', 'red'])
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    ax.set_ylim(0.6, .9)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center')

plt.tight_layout()
plt.savefig("docs/metrics_comparison.png")
plt.show()

# %% [markdown]
# From the visualization above, we can see that SVM is the best model for red wine classification. It surpass the others model in every matrics.


