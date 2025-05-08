#!/usr/bin/env python3
import subprocess
import sys

def main():
    # First, check if the required packages are installed
    try:
        import pandas
        import numpy
        import matplotlib
        import sklearn
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install the required packages using:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the matcher with the test data
    print("\nRunning matcher on test data...")
    cmd = [
        sys.executable, "matcher.py", 
        "--input", "test_data.csv", 
        "--output", "matches.csv", 
        "--iterations", "5000",
        "--preference-weight", "0.6",
        "--distance-weight", "0.4",
        "--objective", "minimize_sum"
    ]
    
    # Add visualization if matplotlib is properly set up
    try:
        import matplotlib.pyplot
        cmd.append("--visualize")
        print("Visualization enabled")
    except:
        print("Visualization disabled (matplotlib backend issue)")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\nMatching completed successfully!")
        print("Check matches.csv for the results.")
    except subprocess.CalledProcessError as e:
        print(f"Error running matcher: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()