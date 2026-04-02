from common_imports import *

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------- FIXED IQR ----------------------
class IQRCapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        self.lower_bounds_ = Q1 - 1.5 * IQR
        self.upper_bounds_ = Q3 + 1.5 * IQR

        return self

    def transform(self, X):

        is_array = False
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            is_array = True

        X = X.clip(self.lower_bounds_, self.upper_bounds_, axis=1)

        return X.values if is_array else X

    # ✅ ADD THIS METHOD (IMPORTANT)
    def get_feature_names_out(self, input_features=None):
        return input_features


# ---------------------- PREPROCESS ----------------------
def preprocess_data(df):
    df = df.drop_duplicates()
    y = df['HeartDisease'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['HeartDisease'])
    return X, y


# ---------------------- SAMPLING ----------------------
def stratified_sample(df, n=10000):
    if len(df) > n:
        df_yes = df[df['HeartDisease'] == 'Yes'].sample(n//2, random_state=42)
        df_no = df[df['HeartDisease'] == 'No'].sample(n//2, random_state=42)
        df = pd.concat([df_yes, df_no]).reset_index(drop=True)
    return df

# def stratified_sample(df, n=10000):
#     if len(df) > n:
#         df = df.sample(n=n, random_state=42)
#     return df


# ---------------------- CHI-SQUARE ----------------------
def chi_square_selection(X, y):
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    selected = []

    for col in cat_cols:
        table = pd.crosstab(X[col], y)
        _, p, _, _ = chi2_contingency(table)
        if p < 0.05:
            selected.append(col)

    return selected


# ---------------------- PREPROCESSOR ----------------------
def build_preprocessor(X, cat_cols):

    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("iqr", IQRCapper()),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])


# ---------------------- LASSO ----------------------
def lasso_selection(preprocessor, X, y):
    pipe = Pipeline([
        ("prep", preprocessor),
        ("lasso", LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5))
    ])
    pipe.fit(X, y)

    features = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["lasso"].coef_[0]

    df = pd.DataFrame({"Feature": features, "Coef": coefs})
    return df[df["Coef"] != 0]


# ---------------------- MODELS ----------------------
def get_models():
    return {

        "Logistic": LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced'
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),

        "DecisionTree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            criterion='gini',
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2  # Euclidean
        ),

        "SVM": SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced'
        ),

        "NaiveBayes": GaussianNB(
            var_smoothing=1e-9
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,   # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            scale_pos_weight=1,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
    }
# ---------------------- METRICS ----------------------
def print_metrics(y_true, y_pred, y_proba, label):

    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n{label} Classification Report:")
    print(classification_report(y_true, y_pred))

    print(f"{label} Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"{label} F1: {f1:.4f}")
    print(f"{label} Recall: {recall:.4f}")
    print(f"{label} ROC-AUC: {auc:.4f}")

    return f1, recall, auc


# ---------------------- TRAIN ----------------------
# ---------------------- TRAIN ----------------------
def train_models(preprocessor, X_train, y_train, X_test, y_test):

    models = get_models()
    results = {}

    for name, model in models.items():

        print("\n" + "="*70)
        print(f"MODEL: {name}")
        print("="*70)

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        try:
            pipe.fit(X_train, y_train)

            # Train
            y_train_pred = pipe.predict(X_train)
            y_train_proba = pipe.predict_proba(X_train)[:, 1]

            # Test
            y_test_pred = pipe.predict(X_test)
            y_test_proba = pipe.predict_proba(X_test)[:, 1]

            # -------- METRICS --------
            train_f1, _, _ = print_metrics(
                y_train, y_train_pred, y_train_proba, "TRAIN"
            )

            test_f1, test_recall, test_auc = print_metrics(
                y_test, y_test_pred, y_test_proba, "TEST"
            )

            # ---------------- OVERFITTING CHECK ----------------
            overfit_penalty = abs(train_f1 - test_f1)

            print(f"\nOverfitting (F1 diff): {overfit_penalty:.4f}")

            # ---------------- FINAL SCORE ----------------
            score = (
                0.4 * test_f1 +
                0.4 * test_recall +
                0.2 * test_auc
            ) - 0.2 * overfit_penalty   # ✅ PENALTY ADDED

            results[name] = {
                "model": pipe,
                "score": score,
                "test_f1": test_f1,
                "test_recall": test_recall,
                "test_auc": test_auc,
                "overfit_penalty": overfit_penalty
            }

        except Exception as e:
            logger.warning(f"{name} failed: {e}")

    # ---------------- BEST MODEL ----------------
    best_name = max(results, key=lambda x: results[x]["score"])
    logger.info(f"Best Model: {best_name}")

    # ---------------------- CREATE RESULTS DATAFRAME ----------------------
    results_df = pd.DataFrame(results).T.reset_index()
    results_df.rename(columns={"index": "Model"}, inplace=True)

    # Sort by score (descending)
    results_df = results_df.sort_values(by="score", ascending=False)

    print("\n\nMODEL COMPARISON TABLE:\n")
    print(results_df[[
        "Model",
        "test_f1",
        "test_recall",
        "test_auc",
        "overfit_penalty",
        "score"
    ]])

    # ---------------------- BEST MODEL ----------------------
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n🏆 BEST MODEL: {best_model_name}")

    return results[best_model_name]["model"], results_df

# ---------------------- MAIN ----------------------
def run_pipeline(path):

    df = pd.read_csv(path)
    logger.info("Loaded data")
        # 🔥 ADD THIS LINE
    df.columns = df.columns.str.strip()

    df = stratified_sample(df)

    X, y = preprocess_data(df)

    cat_cols = chi_square_selection(X, y)

    preprocessor = build_preprocessor(X, cat_cols)

    lasso_features = lasso_selection(preprocessor, X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    best_model, results = train_models(
        preprocessor, X_train, y_train, X_test, y_test
    )

    print("\n🏆 BEST MODEL SELECTED")

    return best_model, results, lasso_features


# ---------------------- RUN ----------------------
if __name__ == "__main__":
    best_model, results, lasso_features = run_pipeline(
        "../data/heart_2020_sample_10k.csv"
    )