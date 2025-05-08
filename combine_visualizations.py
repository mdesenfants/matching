#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def main():
    # Create a new figure for the combined visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Load the annealing process images
    img_sum = mpimg.imread('annealing_minimize_sum.png')
    img_std = mpimg.imread('annealing_minimize_std_dev.png')
    
    # Display the images
    axes[0].imshow(img_sum)
    axes[0].set_title('Minimize Sum Approach')
    axes[0].axis('off')
    
    axes[1].imshow(img_std)
    axes[1].set_title('Minimize Std Dev Approach')
    axes[1].axis('off')
    
    # Add a common title
    fig.suptitle('Simulated Annealing Process Comparison', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.savefig('annealing_comparison.png', dpi=300)
    print("Saved combined visualization to annealing_comparison.png")

if __name__ == "__main__":
    main()