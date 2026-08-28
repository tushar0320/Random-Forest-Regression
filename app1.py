import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Random Forest Regressor",
    page_icon="🌲",
    layout="wide"
)

st.title("🌲 Random Forest Regressor")
st.write("Upload a CSV dataset, choose the target column, train a Random Forest regression model, and make predictions.")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

st.subheader("Dataset")
st.dataframe(df.head(10), use_container_width=True)

st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")

target = st.selectbox("Select the target (value to predict)", df.columns)

if target:
    X = df.drop(columns=[target])
    y = df[target]

    # Convert the target to numeric for regression.
    y = pd.to_numeric(y, errors="coerce")
    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    if len(y) < 5:
        st.error("The target column must contain at least 5 numeric values.")
        st.stop()

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        st.error("No usable feature columns were found.")
        st.stop()

    preprocessor = ColumnTransformer(transformers=transformers)

    st.sidebar.header("Model settings")
    n_estimators = st.sidebar.slider("Number of trees", 50, 500, 100, 10)
    max_depth = st.sidebar.slider("Maximum tree depth", 2, 50, 10)
    test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.sidebar.number_input("Random state", 0, 9999, 42)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    if st.button("🚀 Train Random Forest", type="primary"):
        with st.spinner("Training model..."):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                rmse = mse ** 0.5
                r2 = r2_score(y_test, predictions)

                st.session_state["model"] = model
                st.session_state["feature_columns"] = X.columns.tolist()
                st.session_state["numeric_features"] = numeric_features
                st.session_state["categorical_features"] = categorical_features

                st.subheader("Model performance")
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mae:.4f}")
                c2.metric("RMSE", f"{rmse:.4f}")
                c3.metric("R² Score", f"{r2:.4f}")

                st.success("Model trained successfully!")

            except Exception as e:
                st.error(f"Training failed: {e}")

if "model" in st.session_state:
    st.divider()
    st.subheader("Make a prediction")
    st.write("Enter values for the feature columns below.")

    input_data = {}

    for column in st.session_state["feature_columns"]:
        if column in st.session_state["numeric_features"]:
            series = pd.to_numeric(df[column], errors="coerce")
            default = float(series.median()) if series.notna().any() else 0.0
            input_data[column] = st.number_input(
                column,
                value=default
            )
        else:
            values = df[column].dropna().astype(str).unique().tolist()
            if values:
                input_data[column] = st.selectbox(column, values)
            else:
                input_data[column] = ""

    if st.button("🔮 Predict"):
        input_df = pd.DataFrame([input_data])
        try:
            prediction = st.session_state["model"].predict(input_df)[0]
            st.success(f"Predicted value: **{prediction:.4f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
