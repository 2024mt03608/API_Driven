
def run_basic_eda(df):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import logging

    # ---------------------- LOGGING ----------------------
    logger = logging.getLogger(__name__)

    logger.info("Dataset received for EDA")

    # ---------------------- BASIC INFO ----------------------
    logger.info(f"Shape of dataset: {df.shape}")

    logger.info("First 5 rows:")
    logger.info(f"\n{df.head()}")

    logger.info("Column names:")
    logger.info(f"\n{df.columns.tolist()}")

    # ---------------------- DATA TYPES ----------------------
    logger.info("Data Types:")
    logger.info(f"\n{df.dtypes}")

    # ---------------------- MISSING VALUES ----------------------
    logger.info("Missing Values Count:")
    logger.info(f"\n{df.isnull().sum()}")

    logger.info("Missing Values Percentage:")
    missing_percent = (df.isnull().sum() / len(df)) * 100
    logger.info(f"\n{missing_percent}")

    # ---------------------- DUPLICATES ----------------------
    duplicates = df.duplicated().sum()
    logger.info(f"Number of duplicate rows: {duplicates}")

    # ---------------------- SUMMARY ----------------------
    logger.info("Numerical Summary:")
    logger.info(f"\n{df.describe()}")

    logger.info("Categorical Summary:")
    logger.info(f"\n{df.describe(include='object')}")

    # ---------------------- UNIQUE VALUES ----------------------
    for col in df.columns:
        logger.info(f"{col} - Unique values: {df[col].nunique()}")

    # ---------------------- VALUE COUNTS ----------------------
    for col in df.select_dtypes(include='object').columns:
        logger.info(f"Value counts for {col}:")
        logger.info(f"\n{df[col].value_counts().head()}")

    # ---------------------- DISTRIBUTION ----------------------
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    # ---------------------- TARGET ----------------------
    if 'HeartDisease' in df.columns:
        logger.info("Target variable distribution:")
        logger.info(f"\n{df['HeartDisease'].value_counts()}")

        plt.figure()
        sns.countplot(x='HeartDisease', data=df)
        plt.title("Heart Disease Distribution")
        plt.show()

    return df   # ✅ IMPORTANT