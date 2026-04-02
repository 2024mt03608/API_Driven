def run_eda(df):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    print("Running EDA (optimized)...")

    # ---------------------- SAMPLE (CRITICAL FIX) ----------------------
    df = df.sample(n=10000, random_state=42) if len(df) > 10000 else df

    # Convert object → category (memory optimization)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='category').columns.tolist()

    # Limit columns (avoid too many plots)
    num_cols = num_cols[:8]
    cat_cols = cat_cols[:5]

    # ---------------------- 1. CORRELATION ----------------------
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, mask=mask, annot=False)   # ❌ removed annot (heavy)
        plt.title("Correlation Heatmap")
        plt.show()

    # ---------------------- 2. HISTOGRAM ----------------------
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=False)  # ❌ removed KDE (heavy)
        plt.title(f"{col}")
        plt.show()

    # ---------------------- 3. BOX PLOT ----------------------
    for col in num_cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"{col}")
        plt.show()

    # ---------------------- 4. COUNT PLOT ----------------------
    for col in cat_cols:
        plt.figure()
        sns.countplot(x=df[col])
        plt.xticks(rotation=45)
        plt.title(f"{col}")
        plt.show()

    # ---------------------- 5. SCATTER ----------------------
    if 'HeartDisease' in df.columns:
        for col in num_cols[:5]:
            if col != 'HeartDisease':
                plt.figure()
                sns.scatterplot(x=df[col], y=df['HeartDisease'])
                plt.title(f"{col} vs Target")
                plt.show()

    # ---------------------- 6. PAIRPLOT (VERY HEAVY → LIMIT) ----------------------
    if len(num_cols) >= 2:
        sample_df = df[num_cols].sample(n=500, random_state=42)
        sns.pairplot(sample_df)
        plt.show()

    # ---------------------- 7. TARGET ----------------------
    if 'HeartDisease' in df.columns:
        plt.figure()
        sns.countplot(x='HeartDisease', data=df)
        plt.title("Target Distribution")
        plt.show()

    return df