#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load both result files
    min_sum_df = pd.read_csv('matches.csv')
    min_std_df = pd.read_csv('matches_std_dev.csv')
    
    # Calculate key statistics
    min_sum_total_dist = min_sum_df['euclidean_distance'].sum()
    min_sum_avg_dist = min_sum_df['euclidean_distance'].mean()
    min_sum_std_dev = min_sum_df['euclidean_distance'].std()
    min_sum_avg_pref = min_sum_df['match_score'].mean()
    
    min_std_total_dist = min_std_df['euclidean_distance'].sum()
    min_std_avg_dist = min_std_df['euclidean_distance'].mean()
    min_std_std_dev = min_std_df['euclidean_distance'].std()
    min_std_avg_pref = min_std_df['match_score'].mean()
    
    # Print statistics
    print('=== MINIMIZE SUM OF DISTANCES ===')
    print(f'Total distance: {min_sum_total_dist:.4f}')
    print(f'Average distance: {min_sum_avg_dist:.4f}')
    print(f'Std dev of distances: {min_sum_std_dev:.4f}')
    print(f'Average preference: {min_sum_avg_pref:.2f}')
    print()
    
    print('=== MINIMIZE STD DEV OF DISTANCES ===')
    print(f'Total distance: {min_std_total_dist:.4f}')
    print(f'Average distance: {min_std_avg_dist:.4f}')
    print(f'Std dev of distances: {min_std_std_dev:.4f}')
    print(f'Average preference: {min_std_avg_pref:.2f}')
    
    # Create visualizations
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Distance comparison
    plt.subplot(2, 2, 1)
    plt.bar(['Min Sum', 'Min Std Dev'], [min_sum_total_dist, min_std_total_dist])
    plt.title('Total Euclidean Distance')
    plt.ylabel('Sum of distances')
    
    # Subplot 2: Standard deviation comparison
    plt.subplot(2, 2, 2)
    plt.bar(['Min Sum', 'Min Std Dev'], [min_sum_std_dev, min_std_std_dev])
    plt.title('Standard Deviation of Distances')
    plt.ylabel('Standard deviation')
    
    # Subplot 3: Preference comparison
    plt.subplot(2, 2, 3)
    plt.bar(['Min Sum', 'Min Std Dev'], [min_sum_avg_pref, min_std_avg_pref])
    plt.title('Average Preference Score')
    plt.ylabel('Average score')
    
    # Subplot 4: Histograms of distances
    plt.subplot(2, 2, 4)
    bins = np.linspace(0, 1, 20)
    plt.hist(min_sum_df['euclidean_distance'], bins=bins, alpha=0.5, label='Min Sum')
    plt.hist(min_std_df['euclidean_distance'], bins=bins, alpha=0.5, label='Min Std Dev')
    plt.title('Distribution of Euclidean Distances')
    plt.xlabel('Euclidean distance')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('objective_comparison.png')
    print("Saved comparison to 'objective_comparison.png'")

if __name__ == "__main__":
    main()