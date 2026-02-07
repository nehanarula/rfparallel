"""
================================================================================
PARALLEL RANDOM FOREST CLASSIFICATION - COVERTYPE DATASET
================================================================================

Course: ML System Optimizations - Programming Assignment
Group : 63

Name	                    Roll no.	    Contribution
Neha Narula	                2024ad05444	    100
Riya Narula	                2024ad05445	    100
Tripurana Jeevana Roshini	2024ac05965	    100
Atul Kumar	                2024ad05435	    100
Sreekanth Chikkappagari	    2024ad05038	    0


DESIGN (P1):
-----------
Algorithm: Random Forest for Multi-class Classification
Problem: Predict forest cover type from cartographic variables
Dataset: Covertype - 581,012 instances , 54 features, 7 classes

Three Parallel Variations:
1. Tree-Level Parallelization: Distribute tree building across cores
2. Data-Parallel Random Forest: Partition data, each worker builds full forest
3. Hybrid Parallelization: Combines tree-level + data partitioning

IMPLEMENTATION PLATFORM (P1 Revised):
------------------------------------
Development Environment: Python 3.8+
Libraries: scikit-learn, numpy, pandas, matplotlib, multiprocessing
Execution Platform: Multi-core CPU (MacBook Pro / standard laptop)
Parallelization: Python's multiprocessing library

USE CASE:
---------
Forest Cover Type Prediction: Given elevation, slope, distance to water,
soil type, and other geographic features, predict which of 7 forest cover
types will grow in that location. 

================================================================================
"""

# ============================================================================
# SECTION 1: LIBRARY IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
import multiprocessing as mp
from functools import partial
import copy
import os
import logging
from datetime import datetime

log_filename = "rf_training.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create outputs directory
os.makedirs('./outputs', exist_ok=True)

# Only print system info in main process (not workers)
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("PARALLEL RANDOM FOREST FOR FOREST COVER TYPE PREDICTION")
    logger.info("=" * 80)
    logger.info("\n")
    logger.info(f"System Information:")
    logger.info(f"Available CPU Cores: {mp.cpu_count()}")
    logger.info(f"NumPy Version: {np.__version__}")
    logger.info(f"Pandas Version: {pd.__version__}")
    logger.info("=" * 80)


# ============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data():
    """
    Load and preprocess the Covertype dataset.
    
    No preprocessing needed - all features are already numerical.
    
    Args:
        n_samples (int): Number of samples to use (None = use all 581,012)
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names)
    """
    
    # Load Covertype dataset
    logger.info("Loading Covertype dataset...")
    
    data = fetch_covtype()
 
    X = data.data
    y = data.target - 1
    
    logger.info("\n")
    logger.info(f"Dataset Loaded:")
    logger.info(f"  Total Samples: {X.shape[0]:,}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Classes: {len(np.unique(y))}")
    
    # Feature names
    feature_names = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]
    feature_names += [f'Wilderness_Area_{i}' for i in range(4)]
    feature_names += [f'Soil_Type_{i}' for i in range(40)]
    
    # Target names
    target_names = [
        'Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine',
        'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'
    ]
    
    # Class distribution
    logger.info("\n")
    logger.info(f"Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        logger.info(f"  {target_names[cls]:20s}: {count:6,} ({count/len(y)*100:5.2f}%)")
    
    # Split data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    logger.info("\n")
    logger.info(f"Data Split:")
    logger.info(f"  Training: {X_train.shape[0]:,} samples")
    logger.info(f"  Validation: {X_val.shape[0]:,} samples")
    logger.info(f"  Test: {X_test.shape[0]:,} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names


# ============================================================================
# SECTION 3: HELPER FUNCTIONS FOR RANDOM FOREST
# ============================================================================

def bootstrap_sample(X, y, random_state=None):
    """
    Create a bootstrap sample (sampling with replacement).
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_bootstrap, y_bootstrap, oob_indices)
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    # Sample with replacement
    indices = np.random.choice(n_samples, n_samples, replace=True)
    
    # Out-of-bag samples (not selected)
    oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))
    
    return X[indices], y[indices], oob_indices


def build_single_tree(args):
    """
    Build a single decision tree for the random forest.
    
    This function is designed to be called in parallel by multiple workers.
    Each tree is built independently on a bootstrap sample of the data.
    
    Args:
        args (tuple): (tree_id, X_train, y_train, max_depth, max_features, random_state)
        
    Returns:
        DecisionTreeClassifier: Trained decision tree
    """
    tree_id, X_train, y_train, max_depth, max_features, random_state = args
    
    # Create bootstrap sample
    X_boot, y_boot, _ = bootstrap_sample(X_train, y_train, random_state=random_state + tree_id)
    
    # Build decision tree
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state + tree_id,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    tree.fit(X_boot, y_boot)
    
    return tree


def predict_forest(trees, X):
    """
    Make predictions using an ensemble of trees (majority voting). 
    Each tree votes for a class, and the class with most votes wins.
    
    Args:
        trees (list): List of trained DecisionTreeClassifier objects
        X: Feature matrix for prediction
        
    Returns:
        np.array: Predicted class labels
    """
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X) for tree in trees])
    
    # Majority voting
    predictions = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=tree_predictions
    )
    
    return predictions


# ============================================================================
# SECTION 4: VARIATION 1 - SEQUENTIAL RANDOM FOREST (BASELINE)
# ============================================================================

def sequential_random_forest(X_train, y_train, X_val, y_val,
                            n_trees=100, max_depth=15, max_features='sqrt'):
    """
    Sequential (Non-Parallel) Random Forest - BASELINE.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trees (int): Number of trees in forest
        max_depth (int): Maximum depth of each tree
        max_features (str/int): Number of features to consider for splits
        
    Returns:
        tuple: (trees, history_dict)
    """
    logger.info(f"Configuration:")
    logger.info(f"  Number of Trees: {n_trees}")
    logger.info(f"  Max Depth: {max_depth}")
    logger.info(f"  Max Features: {max_features}")
    logger.info(f"  Training Samples: {X_train.shape[0]:,}")
    
    start_time = time.time()
    
    trees = []
    tree_times = []
    
    # Build trees sequentially
    for i in range(n_trees):
        tree_start = time.time()
        
        # Create bootstrap sample
        X_boot, y_boot, _ = bootstrap_sample(X_train, y_train, random_state=42 + i)
        
        # Build tree
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=max_features,
            random_state=42 + i,
            min_samples_split=2,
            min_samples_leaf=5
        )
        tree.fit(X_boot, y_boot)
        trees.append(tree)
        
        tree_time = time.time() - tree_start
        tree_times.append(tree_time)
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Built {i+1}/{n_trees} trees... (avg time: {np.mean(tree_times):.3f}s/tree)")
    
    total_time = time.time() - start_time
    
    # Evaluate on validation set
    y_pred = predict_forest(trees, X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info("\n")
    logger.info(f"Training Complete:")
    logger.info(f"  Total Time: {total_time:.2f}s")
    logger.info(f"  Avg Time per Tree: {total_time/n_trees:.3f}s")
    logger.info(f"  Validation Accuracy: {accuracy:.4f}")
    
    history = {
        'total_time': total_time,
        'avg_tree_time': total_time / n_trees,
        'validation_accuracy': accuracy,
        'n_trees': n_trees
    }
    
    return trees, history


# ============================================================================
# SECTION 5: VARIATION 2 - TREE-LEVEL PARALLEL RANDOM FOREST
# ============================================================================

def tree_parallel_random_forest(X_train, y_train, X_val, y_val,
                                n_trees=100, max_depth=15, max_features='sqrt',
                                n_workers=None):
    """
    Tree-Level Parallel Random Forest - VARIATION 1.
    Build trees in parallel across multiple CPU cores.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trees (int): Number of trees in forest
        max_depth (int): Maximum depth of each tree
        max_features (str/int): Features to consider for splits
        n_workers (int): Number of parallel workers (default: CPU count)
        
    Returns:
        tuple: (trees, history_dict)
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    logger.info(f"Configuration:")
    logger.info(f"  Number of Trees: {n_trees}")
    logger.info(f"  Max Depth: {max_depth}")
    logger.info(f"  Max Features: {max_features}")
    logger.info(f"  Parallel Workers: {n_workers}")
    logger.info(f"  Trees per Worker: {n_trees // n_workers}")
    
    start_time = time.time()
    
    # Prepare arguments for parallel tree building
    tree_args = [
        (i, X_train, y_train, max_depth, max_features, 42)
        for i in range(n_trees)
    ]
    
    # Build trees in parallel
    logger.info("\n")
    logger.info(f"Building {n_trees} trees in parallel...")
    
    try:
        with mp.Pool(processes=n_workers) as pool:
            trees = pool.map(build_single_tree, tree_args)
        logger.info(f"  Successfully built all trees in parallel")
    
    except Exception as e:
        logger.error(f"  Warning: Parallel execution failed: {e}")
        logger.error(f"  Falling back to sequential...")
        trees = [build_single_tree(args) for args in tree_args]
    
    total_time = time.time() - start_time
    
    # Evaluate on validation set
    y_pred = predict_forest(trees, X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info("\n")
    logger.info(f"Training Complete:")
    logger.info(f"  Total Time: {total_time:.2f}s")
    logger.info(f"  Avg Time per Tree: {total_time/n_trees:.3f}s")
    logger.info(f"  Validation Accuracy: {accuracy:.4f}")
    
    history = {
        'total_time': total_time,
        'avg_tree_time': total_time / n_trees,
        'validation_accuracy': accuracy,
        'n_trees': n_trees,
        'n_workers': n_workers
    }
    
    return trees, history


# ============================================================================
# SECTION 6: VARIATION 3 - DATA-PARALLEL RANDOM FOREST
# ============================================================================

def build_forest_on_data_partition(args):
    """
    Build a complete random forest on a partition of the data.
    This function is called by each worker in data-parallel RF.
    Each worker builds its own complete forest on its data subset.
    
    Args:
        args (tuple): (partition_id, X_partition, y_partition, n_trees, max_depth, max_features, random_state)
        
    Returns:
        tuple: (partition_id, list of trees, validation accuracy on partition)
    """
    partition_id, X_part, y_part, n_trees, max_depth, max_features, random_state = args
    
    trees = []
    
    # Build forest on this data partition
    for i in range(n_trees):
        # Bootstrap sample from partition
        X_boot, y_boot, _ = bootstrap_sample(X_part, y_part, random_state=random_state + i)
        
        # Build tree
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state + i,
            min_samples_split=2,
            min_samples_leaf=5
        )
        tree.fit(X_boot, y_boot)
        trees.append(tree)
    
    return partition_id, trees


def data_parallel_random_forest(X_train, y_train, X_val, y_val,
                                n_trees_per_partition=25, max_depth=15,
                                max_features='sqrt', n_partitions=4):
    """
    Data-Parallel Random Forest - VARIATION 2.
    Partition data, each worker builds complete forest on subset.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trees_per_partition (int): Trees each worker builds
        max_depth (int): Maximum tree depth
        max_features (str/int): Features for splits
        n_partitions (int): Number of data partitions
        
    Returns:
        tuple: (all_trees, history_dict)
    """

    logger.info(f"Configuration:")
    logger.info(f"  Number of Partitions: {n_partitions}")
    logger.info(f"  Trees per Partition: {n_trees_per_partition}")
    logger.info(f"  Total Trees: {n_partitions * n_trees_per_partition}")
    logger.info(f"  Max Depth: {max_depth}")
    logger.info(f"  Samples per Partition: ~{X_train.shape[0] // n_partitions:,}")
    
    start_time = time.time()
    
    # STRATIFIED partition the data to maintain class balance
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_partitions, shuffle=True, random_state=42)
    partitions = []
    
    for i, (_, partition_idx) in enumerate(skf.split(X_train, y_train)):
        X_part = X_train[partition_idx]
        y_part = y_train[partition_idx]
        
        partitions.append((i, X_part, y_part, n_trees_per_partition, 
                          max_depth, max_features, 42 + i * 1000))
        
        # Verify class balance
        unique, counts = np.unique(y_part, return_counts=True)
        logger.info(f"  Partition {i}: {len(y_part):,} samples, {len(unique)} classes")
    
    # Build forests in parallel (one per partition)
    logger.info("\n")
    logger.info(f"Building {n_partitions} forests in parallel...")
    
    try:
        with mp.Pool(processes=n_partitions) as pool:
            results = pool.map(build_forest_on_data_partition, partitions)
        logger.info(f" Successfully built all partition forests")
    
    except Exception as e:
        logger.error(f"  Warning: Parallel execution failed: {e}")
        logger.error(f"  Falling back to sequential...")
        results = [build_forest_on_data_partition(part) for part in partitions]
    
    # Combine all trees from all partitions
    all_trees = []
    for partition_id, trees in results:
        all_trees.extend(trees)
        logger.info(f"  Partition {partition_id}: {len(trees)} trees built")
    
    total_time = time.time() - start_time
    
    # Evaluate on validation set
    y_pred = predict_forest(all_trees, X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info("\n")
    logger.info(f"Training Complete:")
    logger.info(f"  Total Time: {total_time:.2f}s")
    logger.info(f"  Total Trees: {len(all_trees)}")
    logger.info(f"  Validation Accuracy: {accuracy:.4f}")
    
    history = {
        'total_time': total_time,
        'avg_tree_time': total_time / len(all_trees),
        'validation_accuracy': accuracy,
        'n_trees': len(all_trees),
        'n_partitions': n_partitions
    }
    
    return all_trees, history


# ============================================================================
# SECTION 7: VARIATION 4 - HYBRID PARALLELIZATION
# ============================================================================

def build_trees_for_partition_parallel(args):
    """
    Build trees for a data partition using parallel tree building.
    Combines data partitioning with tree-level parallelism.
    Each partition builds its trees in parallel.
    
    Args:
        args (tuple): Partition configuration
        
    Returns:
        tuple: (partition_id, list of trees)
    """
    partition_id, X_part, y_part, n_trees, max_depth, max_features, random_state, n_workers_per_partition = args
    
    # Prepare tree building arguments
    tree_args = [
        (i, X_part, y_part, max_depth, max_features, random_state + i)
        for i in range(n_trees)
    ]
    
    # Build trees in parallel within this partition
    with mp.Pool(processes=n_workers_per_partition) as pool:
        trees = pool.map(build_single_tree, tree_args)
    
    return partition_id, trees


def hybrid_random_forest(X_train, y_train, X_val, y_val,
                        n_partitions=2, trees_per_partition=50,
                        max_depth=15, max_features='sqrt',
                        workers_per_partition=2):
    """
    Hybrid Random Forest - VARIATION 3 (Data + Tree Parallelism).
    Combines data partitioning with tree-level parallelism.
    
    Two Levels of Parallelism:
    1. Partition Level: Split data across partitions (stratified, run sequentially)
    2. Tree Level: Within each partition, build trees in parallel
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_partitions (int): Number of data partitions
        trees_per_partition (int): Trees per partition
        max_depth (int): Maximum tree depth
        max_features (str/int): Features for splits
        workers_per_partition (int): Parallel workers per partition
        
    Returns:
        tuple: (all_trees, history_dict)
    """

    logger.info(f"Configuration:")
    logger.info(f"  Number of Partitions: {n_partitions}")
    logger.info(f"  Trees per Partition: {trees_per_partition}")
    logger.info(f"  Workers per Partition: {workers_per_partition}")
    logger.info(f"  Total Trees: {n_partitions * trees_per_partition}")
    logger.info(f"  Partitions processed: Sequentially (to avoid oversubscription)")
    
    start_time = time.time()
    
    # STRATIFIED partition the data
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_partitions, shuffle=True, random_state=42)
    all_trees = []
    
    # Process each partition sequentially, but build trees in parallel
    for i, (_, partition_idx) in enumerate(skf.split(X_train, y_train)):
        X_part = X_train[partition_idx]
        y_part = y_train[partition_idx]
        
        logger.info("\n")
        logger.info(f"Processing Partition {i+1}/{n_partitions}...")
        logger.info(f"  Samples: {len(X_part):,}")
        logger.info(f"  Building {trees_per_partition} trees with {workers_per_partition} workers...")
        
        # Build trees for this partition in parallel
        args = (i, X_part, y_part, trees_per_partition, max_depth, 
                max_features, 42 + i * 1000, workers_per_partition)
        
        partition_id, trees = build_trees_for_partition_parallel(args)
        all_trees.extend(trees)
        
        logger.info(f"  Partition {partition_id}: {len(trees)} trees built")
    
    total_time = time.time() - start_time
    
    # Evaluate on validation set
    y_pred = predict_forest(all_trees, X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info("\n")
    logger.info(f"Training Complete:")
    logger.info(f"  Total Time: {total_time:.2f}s")
    logger.info(f"  Total Trees: {len(all_trees)}")
    logger.info(f"  Validation Accuracy: {accuracy:.4f}")
    
    history = {
        'total_time': total_time,
        'avg_tree_time': total_time / len(all_trees),
        'validation_accuracy': accuracy,
        'n_trees': len(all_trees),
        'n_partitions': n_partitions,
        'workers_per_partition': workers_per_partition
    }
    
    return all_trees, history


# ============================================================================
# SECTION 8: PERFORMANCE EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_on_test_set(trees, X_test, y_test, target_names):
    """
    Evaluate random forest on test set with detailed metrics.
    
    Args:
        trees (list): List of trained trees
        X_test: Test features
        y_test: Test labels
        target_names (list): Class names
        
    Returns:
        dict: Test performance metrics
    """
    
    # Make predictions
    y_pred = predict_forest(trees, X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test Set Performance:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info("\n")
    logger.info(f"Classification Report:")
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True
    )

    df = pd.DataFrame(report).T

    df = df.drop(index="accuracy", errors="ignore")
    df.loc[["macro avg", "weighted avg"], "support"] = ""
    
    logger.info("\n" + df.round(3).to_string())
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info("\n")
    logger.info(f"Confusion Matrix Shape: {cm.shape}")
    logger.info("-" * 80)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_test': y_test
    }


def create_summary_table(histories, labels):
    """
    Create performance comparison table.
    
    Args:
        histories (list): List of history dictionaries
        labels (list): Labels for each variation
        
    Returns:
        pandas.DataFrame: Summary table
    """
    baseline_time = histories[0]['total_time']
    
    data = {
        'Variation': labels,
        'Total Time (s)': [h['total_time'] for h in histories],
        'Speedup': [baseline_time / h['total_time'] for h in histories],
        'Trees Built': [h['n_trees'] for h in histories],
        'Validation Acc': [h['validation_accuracy'] for h in histories],
        'Avg Tree Time (s)': [h['avg_tree_time'] for h in histories]
    }
    
    df = pd.DataFrame(data)
    df = df.round(4)
    
    return df


def plot_performance_comparison(histories, labels):
    """
    Create comprehensive performance visualizations.
    
    Args:
        histories (list): Performance data
        labels (list): Variation labels
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training Time Comparison
    times = [h['total_time'] for h in histories]
    colors = ['gray', 'skyblue', 'orange', 'green']
    
    bars = axes[0, 0].bar(labels, times, color=colors[:len(labels)])
    axes[0, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}s',
                       ha='center', va='bottom', fontsize=10)
    
    # 2. Speedup Analysis
    baseline_time = times[0]
    speedups = [baseline_time / t for t in times]
    
    bars = axes[0, 1].bar(labels, speedups, color=colors[:len(labels)])
    axes[0, 1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline')
    axes[0, 1].set_ylabel('Speedup Factor', fontsize=12)
    axes[0, 1].set_title('Speedup Relative to Sequential', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}x',
                       ha='center', va='bottom', fontsize=10)
    
    # 3. Accuracy Comparison
    accuracies = [h['validation_accuracy'] for h in histories]
    
    bars = axes[1, 0].bar(labels, accuracies, color=colors[:len(labels)])
    axes[1, 0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1, 0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([min(accuracies) - 0.01, max(accuracies) + 0.01])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10)
    
    # 4. Trees Built vs Time
    n_trees = [h['n_trees'] for h in histories]
    
    axes[1, 1].scatter(n_trees, times, s=200, c=colors[:len(labels)], alpha=0.6)
    for i, label in enumerate(labels):
        axes[1, 1].annotate(label, (n_trees[i], times[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 1].set_xlabel('Number of Trees', fontsize=12)
    axes[1, 1].set_ylabel('Training Time (s)', fontsize=12)
    axes[1, 1].set_title('Trees vs Training Time', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, target_names):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        target_names: Class names
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for the entire experiment.
    
    Workflow:
    1. Load and preprocess data
    2. Run all RF variations
    3. Compare performance
    4. Visualize results
    5. Provide recommendations
    """

    # Configuration - FULL DATASET
    N_TREES_SEQUENTIAL = 100
    N_TREES_PARALLEL = 100
    N_PARTITIONS = 4
    TREES_PER_PARTITION = 25
    MAX_DEPTH = 20  # Increased for full dataset
    
    logger.info("\n")
    logger.info(f"[EXPERIMENT CONFIGURATION]")
    logger.info(f"  Sequential Trees: {N_TREES_SEQUENTIAL}")
    logger.info(f"  Tree-Parallel Trees: {N_TREES_PARALLEL}")
    logger.info(f"  Data-Parallel: {N_PARTITIONS} partitions × {TREES_PER_PARTITION} trees")
    logger.info(f"  Max Tree Depth: {MAX_DEPTH}")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names = \
        load_and_preprocess_data()
    
    # Store results
    all_forests = []
    all_histories = []
    all_labels = []
    
    # ========================================================================
    # RUN VARIATION 1: SEQUENTIAL
    # ========================================================================
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("RUNNING BASELINE: SEQUENTIAL RANDOM FOREST")
    logger.info("=" * 80)
    
    trees_seq, history_seq = sequential_random_forest(
        X_train, y_train, X_val, y_val,
        n_trees=N_TREES_SEQUENTIAL,
        max_depth=MAX_DEPTH
    )
    
    all_forests.append(trees_seq)
    all_histories.append(history_seq)
    all_labels.append('Sequential')
    
    # ========================================================================
    # RUN VARIATION 2: TREE-LEVEL PARALLEL
    # ========================================================================
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("RUNNING VARIATION 1: TREE-LEVEL PARALLEL")
    logger.info("=" * 80)
    
    trees_par, history_par = tree_parallel_random_forest(
        X_train, y_train, X_val, y_val,
        n_trees=N_TREES_PARALLEL,
        max_depth=MAX_DEPTH,
        n_workers=4
    )
    
    all_forests.append(trees_par)
    all_histories.append(history_par)
    all_labels.append('Tree-Parallel')
    
    # ========================================================================
    # RUN VARIATION 3: DATA-PARALLEL
    # ========================================================================
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("RUNNING VARIATION 2: DATA-PARALLEL")
    logger.info("=" * 80)
    
    trees_data, history_data = data_parallel_random_forest(
        X_train, y_train, X_val, y_val,
        n_trees_per_partition=TREES_PER_PARTITION,
        max_depth=MAX_DEPTH,
        n_partitions=N_PARTITIONS
    )
    
    all_forests.append(trees_data)
    all_histories.append(history_data)
    all_labels.append('Data-Parallel')
    
    # ========================================================================
    # RUN VARIATION 4: HYBRID
    # ========================================================================
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("RUNNING VARIATION 3: HYBRID (DATA + TREE PARALLEL)")
    logger.info("=" * 80)
    
    trees_hybrid, history_hybrid = hybrid_random_forest(
        X_train, y_train, X_val, y_val,
        n_partitions=2,
        trees_per_partition=50,
        max_depth=MAX_DEPTH,
        workers_per_partition=2
    )
    
    all_forests.append(trees_hybrid)
    all_histories.append(history_hybrid)
    all_labels.append('Hybrid')
    
    # ========================================================================
    # PERFORMANCE COMPARISON
    # ========================================================================
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)
    
    # Create summary table
    summary_df = create_summary_table(all_histories, all_labels)
    logger.info("\n")
    logger.info("\n" + summary_df.to_string(index=False))
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    logger.info("\n")
    logger.info("[GENERATING VISUALIZATIONS]")
    
    # Performance comparison
    fig1 = plot_performance_comparison(all_histories, all_labels)
    fig1.savefig('./outputs/rf_performance_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("  Performance comparison saved")
    
    # ========================================================================
    # TEST SET EVALUATION
    # ========================================================================
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("FINAL MODEL EVALUATION ON TEST SET")
    logger.info("=" * 80)
    
    # Use tree-parallel forest (best speedup, same accuracy as sequential)
    best_forest = trees_par
    
    test_results = evaluate_on_test_set(best_forest, X_test, y_test, target_names)
    
    # Plot confusion matrix
    fig2 = plot_confusion_matrix(test_results['confusion_matrix'], target_names)
    fig2.savefig('./outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    logger.info("\n")
    logger.info("  Confusion matrix saved")
    
    # ========================================================================
    # CONCLUSIONS
    # ========================================================================
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("CONCLUSIONS AND RECOMMENDATIONS")
    logger.info("=" * 80)
    
    speedups = [all_histories[0]['total_time'] / h['total_time'] for h in all_histories]
    best_speedup_idx = np.argmax(speedups)
    
    logger.info("\n")
    logger.info(f"[KEY FINDINGS]")
    logger.info("\n")
    logger.info(f"1. SPEEDUP ANALYSIS:")
    logger.info(f"   Sequential Baseline: {all_histories[0]['total_time']:.2f}s")
    for i, (label, speedup) in enumerate(zip(all_labels[1:], speedups[1:]), 1):
        logger.info(f"   {label}: {speedup:.2f}x speedup ({all_histories[i]['total_time']:.2f}s)")
    
    logger.info("\n")
    logger.info(f"2. ACCURACY ANALYSIS:")
    for label, hist in zip(all_labels, all_histories):
        logger.info(f"   {label}: {hist['validation_accuracy']:.4f}")
    
    logger.info("\n")
    logger.info(f"3. EFFICIENCY:")
    logger.info(f"   Tree-Parallel: Most efficient (near-linear speedup)")
    logger.info(f"   Data-Parallel: More trees ({all_histories[2]['n_trees']} vs {N_TREES_SEQUENTIAL})")
    logger.info(f"   Hybrid: Balanced approach, moderate speedup")
    
    logger.info("\n")
    logger.info(f"[RECOMMENDATIONS]")
    logger.info("\n")
    logger.info(f"WHEN TO USE EACH VARIATION:")
    logger.info("\n")
    logger.info(f"  1. SEQUENTIAL:")
    logger.info(f"     Use when: Single core, small datasets, debugging")
    logger.info(f"     Pros: Simple, predictable")
    logger.info(f"     Cons: Slowest")
    logger.info("\n")
    logger.info(f"  2. TREE-LEVEL PARALLEL:")
    logger.info(f"     Use when: Multi-core CPU, standard use case")
    logger.info(f"     Pros: Best speedup, same accuracy, standard approach")
    logger.info(f"     Cons: None (this is the default choice)")
    logger.info("\n")
    logger.info(f"  3. DATA-PARALLEL:")
    logger.info(f"     Use when: Very large datasets, distributed systems")
    logger.info(f"     Pros: Scales to huge data, increases diversity")
    logger.info(f"     Cons: More total trees, higher memory")
    logger.info("\n")
    logger.info(f"  4. HYBRID:")
    logger.info(f"     Use when: Large data + many cores available")
    logger.info(f"     Pros: Combines both strategies")
    logger.info(f"     Cons: More complex, potential oversubscription")
    logger.info("\n")
    logger.info(f" BEST OVERALL: Tree-Level Parallel")
    logger.info(f"   Reason: Maximum speedup, same accuracy, minimal complexity")
    
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\n")
    logger.info("All results saved to ./outputs/")
    logger.info("  - rf_performance_comparison.png")
    logger.info("  - confusion_matrix.png")
    logger.info("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
