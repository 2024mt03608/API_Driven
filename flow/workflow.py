from prefect import flow, task
from BasicStats import run_basic_eda
from Main import (
    run_pipeline,
    stratified_sample,
    preprocess_data,
    chi_square_selection,
    build_preprocessor,
    lasso_selection
)

from common_imports import pd

# ---------------- TASKS ----------------

@task
def load_data(path):
    df = pd.read_csv(path)

    # 🔥 ADD THIS LINE (THIS WILL FIX YOUR ERROR)
    df.columns = df.columns.str.strip()

    print("Columns:", df.columns.tolist())  # debug once

    return df


@task
def basic_stats_task(df):
    df = run_basic_eda(df)
    return df


@task
def eda_task(df):
    from EDA import run_eda
    df = run_eda(df)
    return df


@task
def sample_task(df):
    return stratified_sample(df)

@task
def preprocess_task(df):
    print("Before preprocess columns:", df.columns.tolist())  # 👈 ADD THIS
    return preprocess_data(df)

# @task
# def preprocess_task(df):
#     return preprocess_data(df)


@task
def feature_selection_task(X, y):
    return chi_square_selection(X, y)


@task
def preprocessor_task(X, cat_cols):
    return build_preprocessor(X, cat_cols)


@task
def lasso_task(preprocessor, X, y):
    return lasso_selection(preprocessor, X, y)


@task
def full_training_task(path):
    # fallback: full pipeline
    return run_pipeline(path)


# ---------------- FLOW ----------------
@flow(name="heart-disease-advanced-flow", log_prints=True)
def ml_workflow():

    path = "../data/heart_2020_sample_10k.csv"

    # ---------------- LOAD ORIGINAL ----------------
    df = load_data(path)

    # ---------------- EDA BRANCH ----------------
    _ = basic_stats_task(df)
    _ = eda_task(df)
    # 👉 ignore output (EDA only for viewing)

    # ---------------- MODELING BRANCH ----------------
    df_model = load_data(path)   # ✅ reload fresh data

    df_model = sample_task(df_model)

    X, y = preprocess_task(df_model)

    cat_cols = feature_selection_task(X, y)

    preprocessor = preprocessor_task(X, cat_cols)

    lasso_features = lasso_task(preprocessor, X, y)

    from sklearn.model_selection import train_test_split
    from Main import train_models

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    result, _ = train_models(
        preprocessor, X_train, y_train, X_test, y_test
    )

    print("✅ Workflow Completed")

    return result

# ---------------- DEPLOY ----------------

if __name__ == "__main__":
    ml_workflow.serve(
        name="heart-disease-eda-training",
        tags=["mlops", "eda", "training"],
        interval=180  # every 3 minutes
    )