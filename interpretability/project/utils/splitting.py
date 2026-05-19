"""Dataset splitting utilities for train/validation/test partitioning.

This module provides functions for splitting datasets with support for:
- Random splitting
- Stratified splitting (preserving class distributions)
- Hierarchical clustered splitting (using MMseqs2 cluster similarity thresholds)
"""
import polars as pl

from project.utils.strs import SEED

def hierarchical_clustered_split(
    df: pl.DataFrame,
    val_threshold: str = 'mmseqs_0.50', # High identity for Train/Validation split (e.g., 50% ID)
    test_threshold: str = 'mmseqs_0.25', # Low identity for Train/Val vs Test split (e.g., 25% ID)
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = SEED
):
    """
    Splits a Polars DataFrame into Train, Validation, and Test sets based on hierarchical
    MMseqs2 cluster similarity thresholds to control evolutionary distance.

    Args:
        df: Polars DataFrame containing protein data and MMseqs2 cluster columns.
        val_threshold: The sequence identity cutoff (as a float) used for the HIGH similarity split.
        test_threshold: The sequence identity cutoff (as a float) used for the LOW similarity split.
        val_ratio: Desired proportion for the Validation set (out of the remaining data).
        test_ratio: Desired proportion for the Test set (out of the total data).
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (train_df, val_df, test_df) as Polars DataFrames.
    """
    from sklearn.model_selection import train_test_split

    # Convert to Pandas for sklearn's train_test_split
    df_pandas = df.to_pandas()
    
    # Split Clusters into (Train/Val Pool) and (Test Pool) using the LOW threshold (e.g., 0.30) (high stringency for similarity)
    
    # Get unique cluster representative IDs for the test threshold
    test_reps= df_pandas[test_threshold].unique().tolist()
    
    # Randomly split the LOW-similarity cluster representative IDs
    train_val_reps, test_reps_split = train_test_split(
        test_reps,
        test_size=test_ratio,
        random_state=seed
    )
    
    # Assign the split:
    test_df_pandas = df_pandas[df_pandas[test_threshold].isin(test_reps_split)].copy()
    train_val_df_pandas = df_pandas[df_pandas[test_threshold].isin(train_val_reps)].copy()

    # Split the (Train/Val Pool) into Train and Validation using the HIGH threshold (e.g., 0.50)
    
    # Get unique cluster representative IDs for the HIGH threshold in the remaining data
    val_cluster_reps = train_val_df_pandas[val_threshold].unique().tolist()

    # Calculate adjusted test_size for val_ratio relative to the remaining data
    # This prevents the sum of ratios from exceeding 1.0
    val_test_size_adjusted = val_ratio / (1.0 - test_ratio)
    
    # Randomly split the HIGH-similarity cluster representative IDs
    train_cluster_reps, val_cluster_reps_split = train_test_split(
        val_cluster_reps,
        test_size=val_test_size_adjusted,
        random_state=seed
    )
    
    # Assign the split:
    val_df_pandas = train_val_df_pandas[train_val_df_pandas[val_threshold].isin(val_cluster_reps_split)].copy()
    train_df_pandas = train_val_df_pandas[train_val_df_pandas[val_threshold].isin(train_cluster_reps)].copy()
    
    # Annotate the split column
    train_df_pandas['split'] = 'train'
    val_df_pandas['split'] = 'val'
    test_df_pandas['split'] = 'test'

    # 4. Convert back to Polars DataFrames and return
    train_df_pl = pl.from_pandas(train_df_pandas)
    val_df_pl = pl.from_pandas(val_df_pandas)
    test_df_pl = pl.from_pandas(test_df_pandas)

    return train_df_pl, val_df_pl, test_df_pl

def train_val_test_split(
    df: pl.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    how: str = 'random', # whether to split randomly or 'stratified'
    stratify_col = 'classes', # If stratifying, what column do we stratify on?
    seed: int = SEED,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Splits a polars DataFrame into train, validation, and test sets.

    Args:
        df: The polars DataFrame to split.
        train_size: Proportion for training set.
        val_size: Proportion for validation set.
        test_size: Proportion for test set.
        how: Splitting method ('random' or 'stratified').
        seed: Random seed for reproducibility.

    Returns:
        tuple: Three polars DataFrames for train, validation, and test sets.
    """
    
    # Convert to pandas for sklearn compatibility
    df_pandas = df.to_pandas()

    val_size_adjusted = val_size / (train_size + val_size)

    if how == 'random':
        from sklearn.model_selection import train_test_split

        train_val_df, test_df = train_test_split(
            df_pandas, test_size=test_size, random_state=seed
        )
        
        train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, random_state=seed)

    elif how == 'stratified':
        from sklearn.model_selection import StratifiedShuffleSplit
        # Get classes with fewer than 3 occurrences
        class_counts = df_pandas[stratify_col].astype(str).value_counts()
        rare_classes = class_counts[class_counts < 3].index.to_list()
        
        # Combine rare classes into one 'rare' class
        df_pandas[stratify_col] = df_pandas[stratify_col].astype(str).replace(rare_classes, 'rare')
        
        # Convert to string for consistent stratification (sklearn likes strings/integers)
        y_stratify = df_pandas[stratify_col]

        # First split: train/val and test
        splitter = StratifiedShuffleSplit(n_splits=1, test_size = test_size, random_state=seed)
        train_val_inds, test_inds = next(splitter.split(df_pandas, y=y_stratify))
        train_val_df, test_df = df_pandas.iloc[train_val_inds], df_pandas.iloc[test_inds]
        
        # Second split: train and val
        # Have to make a new splitter with adjusted size
        splitter = StratifiedShuffleSplit(n_splits=1, test_size = val_size_adjusted, random_state=seed)
        # Use the split-specific y for stratification
        train_inds, val_inds = next(splitter.split(train_val_df, y=train_val_df[stratify_col]))
        train_df, val_df = train_val_df.iloc[train_inds], train_val_df.iloc[val_inds]
        
        # Annotate the split column
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
    
    else:
        raise ValueError(f"Invalid split 'how' method: {how}. Must be 'random' or 'stratified'.")


    # Convert back to polars
    train_df = pl.from_pandas(train_df)
    val_df = pl.from_pandas(val_df)
    test_df = pl.from_pandas(test_df)

    return train_df, val_df, test_df




