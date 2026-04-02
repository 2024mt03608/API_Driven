def run_eda(df):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    print("Running EDA...")

    # ---------------------- BASIC SETUP ----------------------
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    # ---------------------- 1. TRIANGULAR CORRELATION ----------------------
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Triangular Correlation Heatmap")
    plt.show()

    # ---------------------- 2. HISTOGRAM ----------------------
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.show()

    # ---------------------- 3. KDE PLOT ----------------------
    for col in num_cols:
        plt.figure()
        sns.kdeplot(df[col], fill=True)
        mean_val = df[col].mean()
        plt.axvline(mean_val, linestyle='--', label=f"Mean: {mean_val:.2f}")
        plt.title(f"KDE Plot of {col}")
        plt.legend()
        plt.show()

    # ---------------------- 4. BOX PLOT ----------------------
    for col in num_cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Box Plot of {col}")
        plt.show()

    # ---------------------- 5. COUNT PLOT ----------------------
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[col])
        plt.title(f"Count Plot of {col}")
        plt.xticks(rotation=45)
        plt.show()

    # ---------------------- 6. SCATTER ----------------------
    if 'HeartDisease' in df.columns:
        for col in num_cols:
            if col != 'HeartDisease':
                plt.figure()
                sns.scatterplot(x=df[col], y=df['HeartDisease'])
                plt.title(f"{col} vs HeartDisease")
                plt.show()

    # ---------------------- 7. PAIRPLOT ----------------------
    sample_df = df.sample(1000) if len(df) > 1000 else df
    sns.pairplot(sample_df[num_cols])
    plt.show()

    # ---------------------- 8. TARGET ----------------------
    if 'HeartDisease' in df.columns:
        plt.figure()
        sns.countplot(x='HeartDisease', data=df)
        plt.title("Target Variable Distribution")
        plt.show()

    return df   # ✅ IMPORTANT (so Prefect chain continues)