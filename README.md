# Prom Date Matcher

A Python application to help a group of platonic friends pick compatible prom dates from among each other using simulated annealing.

## Overview

This tool analyzes survey responses from a group of friends and uses simulated annealing to find optimal prom date pairings while respecting everyone's preferences.

## Features

- Imports survey responses from CSV
- Uses simulated annealing algorithm for matching optimization
- Generates optimal pairing recommendations
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

## Project Structure

```
matcher/
├── README.md
├── requirements.txt
├── matcher.py        # Main application
├── annealing.py      # Simulated annealing implementation
├── data_loader.py    # CSV handling functionality
└── tests/            # Test directory
```