#!/usr/bin/env python3
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def generate_random_matches(male_ids, female_ids):
    """Generate random matches between male and female students."""
    female_ids_shuffled = female_ids.copy()
    random.shuffle(female_ids_shuffled)
    return list(zip(male_ids, female_ids_shuffled))

def calculate_euclidean_distances(matches, feature_vectors):
    """Calculate the Euclidean distances for all matches."""
    distances = []
    for male_id, female_id in matches:
        if male_id in feature_vectors and female_id in feature_vectors:
            dist = np.linalg.norm(feature_vectors[male_id] - feature_vectors[female_id])
            distances.append(dist)
    return distances

def create_feature_vectors(df):
    """Create feature vectors for each student using one-hot encoding."""
    feature_vectors = {}
    categorical_features = ['preferred_activities', 'music_preference', 'personality']
    encoders = {}
    
    # Create encoders for each categorical feature
    for feature in categorical_features:
        unique_values = df[feature].unique()
        unique_values_2d = np.array(unique_values).reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(unique_values_2d)
        encoders[feature] = encoder
    
    # Create feature vector for each student
    for _, student in df.iterrows():
        student_id = student['id']
        
        # Initialize feature vector with numeric features
        feature_vector = [
            student['height'] / 200.0,  # Normalize height
        ]
        
        # Add one-hot encoded categorical features
        for feature in categorical_features:
            value = np.array([[student[feature]]])
            encoded = encoders[feature].transform(value)
            feature_vector.extend(encoded.flatten())
        
        # Store the feature vector
        feature_vectors[student_id] = np.array(feature_vector)
    
    return feature_vectors

def main():
    # Load student data
    df = pd.read_csv('test_data.csv')
    
    # Split by gender
    male_df = df[df['gender'] == 'Male']
    female_df = df[df['gender'] == 'Female']
    
    # Get IDs
    male_ids = male_df['id'].tolist()
    female_ids = female_df['id'].tolist()
    
    # Create feature vectors
    feature_vectors = create_feature_vectors(df)
    
    # Load optimized matches for comparison
    optimized_matches_df = pd.read_csv('matches.csv')
    optimized_male_ids = optimized_matches_df['male_id'].tolist()
    optimized_female_ids = optimized_matches_df['female_name'].tolist()
    optimized_distances = optimized_matches_df['euclidean_distance'].tolist()
    optimized_std_dev = np.std(optimized_distances)
    
    # Generate 1000 random matchings and calculate statistics
    std_devs = []
    avg_distances = []
    all_distances = []  # Store all distance sets for CV calculation
    
    print("Generating 1000 random matchings...")
    for i in range(1000):
        random_matches = generate_random_matches(male_ids, female_ids)
        distances = calculate_euclidean_distances(random_matches, feature_vectors)
        
        std_dev = np.std(distances)
        avg_dist = np.mean(distances)
        
        std_devs.append(std_dev)
        avg_distances.append(avg_dist)
        all_distances.append(distances)
    
    # Calculate statistics
    avg_std_dev = np.mean(std_devs)
    min_std_dev = np.min(std_devs)
    max_std_dev = np.max(std_devs)
    
    # Calculate coefficient of variation for each random matching
    cvs = []
    for i, distances in enumerate(all_distances):
        mean_dist = np.mean(distances)
        std_dev = std_devs[i]
        cv = std_dev / mean_dist if mean_dist > 0 else 0
        cvs.append(cv)
    
    avg_cv = np.mean(cvs)
    min_cv = np.min(cvs)
    max_cv = np.max(cvs)
    
    # Calculate CV for optimized solution
    optimized_mean_dist = np.mean(optimized_distances)
    optimized_cv = optimized_std_dev / optimized_mean_dist if optimized_mean_dist > 0 else 0
    
    # Print results
    print(f"\nStatistics from 1000 random matchings:")
    print(f"Average std dev: {avg_std_dev:.4f}")
    print(f"Min std dev: {min_std_dev:.4f}")
    print(f"Max std dev: {max_std_dev:.4f}")
    print(f"\nAverage coefficient of variation: {avg_cv:.4f}")
    print(f"Min coefficient of variation: {min_cv:.4f}")
    print(f"Max coefficient of variation: {max_cv:.4f}")
    
    print(f"\nOptimized solution:")
    print(f"Standard deviation: {optimized_std_dev:.4f}")
    print(f"Coefficient of variation: {optimized_cv:.4f}")
    
    # Calculate percentile of optimized solution
    percentile = sum(1 for s in std_devs if s > optimized_std_dev) / len(std_devs) * 100
    print(f"Optimized solution is better than {percentile:.1f}% of random matchings")
    
    # Plot histogram of standard deviations
    plt.figure(figsize=(10, 6))
    plt.hist(std_devs, bins=30, alpha=0.7, label='Random Matchings')
    plt.axvline(optimized_std_dev, color='red', linestyle='dashed', 
                linewidth=2, label=f'Optimized Solution ({optimized_std_dev:.4f})')
    plt.xlabel('Standard Deviation of Euclidean Distances')
    plt.ylabel('Frequency')
    plt.title('Comparison of Optimized Solution vs Random Matchings')
    plt.legend()
    plt.savefig('std_dev_comparison.png')
    print("Saved comparison plot to 'std_dev_comparison.png'")

if __name__ == "__main__":
    main()