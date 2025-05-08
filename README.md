# Prom Date Matcher

A Python application to help a group of platonic friends pick compatible prom dates from among each other using simulated annealing.

## Overview

This tool analyzes survey responses from a group of friends and uses simulated annealing to find optimal prom date pairings while respecting everyone's preferences.

## Features

- Imports survey responses from CSV
- Uses simulated annealing algorithm for matching optimization
- Generates optimal pairing recommendations 
- Supports two optimization approaches:
  - Minimize sum of distances (for efficiency)
  - Minimize standard deviation of distances (for equity)
- Includes visualization tools for both approaches
- Considers multiple preference factors

## Requirements

- Python 3.8+
- Required libraries:
  - pandas (for data handling)
  - numpy (for numerical operations)
  - matplotlib (for visualization)

## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/matcher.git
cd matcher

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your survey data in a CSV file
2. Run the matcher:
   ```
   python matcher.py --input survey_data.csv --output matches.csv
   ```

3. For different optimization approaches:
   ```
   # To use the standard deviation minimization approach
   python matcher.py --input survey_data.csv --output matches_std_dev.csv --objective minimize_std_dev
   
   # To use the sum minimization approach (default)
   python matcher.py --input survey_data.csv --output matches_sum.csv --objective minimize_sum
   ```

4. To generate visualizations:
   ```
   # Run with visualization flag
   python matcher.py --input survey_data.csv --output matches.csv --visualize
   
   # Run comparison script
   python compare_objectives.py
   
   # Or use the all-in-one visualization script
   python visualize_annealing.py
   ```

## Project Structure

```
matcher/
├── README.md
├── requirements.txt
├── matcher.py             # Main application
├── annealing.py           # Simulated annealing implementation
├── data_loader.py         # CSV handling functionality
├── compare_objectives.py  # Comparison of optimization approaches
├── visualize_annealing.py # Generate visualizations for both approaches
├── paper.tex              # Research paper (LaTeX version)
├── paper.md               # Research paper (Markdown version)
├── paper.pdf              # Compiled paper with figures
├── objective_comparison.png # Figure comparing objective approaches
├── annealing_comparison.png # Figure comparing annealing processes
└── tests/                 # Test directory
```