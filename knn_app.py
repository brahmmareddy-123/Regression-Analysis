import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# Title
# =========================
st.title("📊 KNN + Decision Tree Regression App")

# =========================
# Upload File
# =========================
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    # =========================
    # Handle Missing Values
    # =========================
    df['income'].fillna(df['income'].mean(), inplace=True)
    df['loan_amount'].fillna(df['loan_amount'].mean(), inplace=True)
    df['credit_score'].fillna(df['credit_score'].mean(), inplace=True)

    # =========================
    # Remove Outliers (IQR)
    # =========================
    def remove_outliers(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[col] >= lower) & (df[col] <= upper)]

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        df = remove_outliers(df, col)

    st.write("Shape after removing outliers:", df.shape)

    # =========================
    # Visualizations
    # =========================
    st.subheader("📦 Boxplots")

    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

    st.subheader("📊 Histograms")

    for col in num_cols:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=30)
        ax.set_title(col)
        st.pyplot(fig)

    # =========================
    # Encoding
    # =========================
    cat_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_cols)

    # =========================
    # Scaling
    # =========================
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # =========================
    # Features & Target
    # =========================
    X = df.drop("target", axis=1)
    y = df["target"]

    # =========================
    # Train-Test Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # K Selection Slider
    # =========================
    k = st.slider("Select K Value", 1, 20, 5)

    # =========================
    # KNN Model
    # =========================
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    knn_train_acc = r2_score(y_train, y_train_pred)
    knn_test_acc = r2_score(y_test, y_test_pred)

    st.subheader("🤖 KNN Results")
    st.write("Train Accuracy:", knn_train_acc)
    st.write("Test Accuracy:", knn_test_acc)

    # =========================
    # Decision Tree
    # =========================
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)

    y_train_pred_dt = dt.predict(X_train)
    y_test_pred_dt = dt.predict(X_test)

    dt_train_acc = r2_score(y_train, y_train_pred_dt)
    dt_test_acc = r2_score(y_test, y_test_pred_dt)

    st.subheader("🌳 Decision Tree Results")
    st.write("Train Accuracy:", dt_train_acc)
    st.write("Test Accuracy:", dt_test_acc)

    # =========================
    # Elbow Plot (R2)
    # =========================
    st.subheader("📈 Elbow Plot (K vs R2)")

    k_values = list(range(1, 21))
    r2_scores = []

    for k_val in k_values:
        model = KNeighborsRegressor(n_neighbors=k_val)
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        r2_scores.append(scores.mean())

    fig, ax = plt.subplots()
    ax.plot(k_values, r2_scores, marker='o')
    ax.set_xlabel("K Value")
    ax.set_ylabel("R2 Score")
    ax.set_title("Elbow Plot")

    st.pyplot(fig)

    # =========================
    # Best K
    # =========================
    best_k = k_values[r2_scores.index(max(r2_scores))]
    st.write("✅ Best K Value:", best_k)

    # =========================
    # Prediction Section
    # =========================
    st.subheader("🔮 Make Prediction")

    input_data = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = knn.predict(input_array)
        st.success(f"Predicted Target: {prediction[0]}")