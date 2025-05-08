#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os

def run_matcher_with_visualization(input_file, output_file, objective, iterations=5000):
    """Run the matcher with visualization enabled and save the figure"""
    # Create figure file name based on objective
    figure_file = f"annealing_{objective}.png"
    
    # Run matcher with visualization flag
    # Use the Python from the virtual environment
    python_executable = "/home/user/projects/matcher/venv/bin/python"
    cmd = [
        python_executable, "matcher.py",
        "--input", input_file,
        "--output", output_file,
        "--objective", objective,
        "--iterations", str(iterations),
        "--visualize"
    ]
    
    # Start process with a new process group so we can terminate it properly
    process = subprocess.Popen(cmd)
    
    try:
        # Wait for process to complete
        process.wait()
        print(f"Generated visualization for {objective} approach")
    except KeyboardInterrupt:
        # If interrupted, make sure to terminate the process
        process.terminate()
        process.wait()
        print("Process interrupted")

def main():
    # Check if test data exists
    if not os.path.exists('test_data.csv'):
        print("Test data not found. Generating...")
        python_executable = "/home/user/projects/matcher/venv/bin/python"
        subprocess.run([python_executable, "generate_test_data.py", "--num-students", "10"])
    
    # Run the matcher with both objectives and visualize
    print("Running matcher with minimize_sum objective...")
    run_matcher_with_visualization(
        input_file='test_data.csv',
        output_file='matches_visual.csv',
        objective='minimize_sum'
    )
    
    # Save current figure
    plt.savefig('annealing_minimize_sum.png')
    plt.close()
    
    print("Running matcher with minimize_std_dev objective...")
    run_matcher_with_visualization(
        input_file='test_data.csv',
        output_file='matches_std_dev_visual.csv',
        objective='minimize_std_dev'
    )
    
    # Save current figure
    plt.savefig('annealing_minimize_std_dev.png')
    plt.close()
    
    print("Visualizations saved as annealing_minimize_sum.png and annealing_minimize_std_dev.png")

if __name__ == "__main__":
    main()