#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import OneHotEncoder

class DataLoader:
    def __init__(self, file_path: str):
        """Initialize the data loader with the path to the CSV file."""
        self.file_path = file_path
        self.data = None
        self.male_students = []
        self.female_students = []
        self.preference_matrix = {}
        self.feature_vectors = {}  # Will store feature vectors for each student
        self.encoders = {}  # Will store one-hot encoders
        
    def load_data(self) -> pd.DataFrame:
        """Load the CSV data into a pandas DataFrame."""
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def prepare_data(self) -> Tuple[List[str], List[str], Dict]:
        """Process the loaded data and extract student IDs and preference matrices."""
        if self.data is None:
            self.load_data()
        
        # Split data by gender
        male_data = self.data[self.data['gender'] == 'Male']
        female_data = self.data[self.data['gender'] == 'Female']
        
        # Extract student IDs
        self.male_students = male_data['id'].tolist()
        self.female_students = female_data['id'].tolist()
        
        # Create initial preference matrix from explicit ratings
        self._create_preference_matrix(male_data, female_data)
        
        # Process categorical features with one-hot encoding
        self._create_feature_vectors()
        
        # Update preference matrix with feature-based compatibility
        self._calculate_euclidean_compatibility()
        
        return self.male_students, self.female_students, self.preference_matrix
    
    def _create_preference_matrix(self, male_data, female_data):
        """Create the initial preference matrix based on explicit ratings."""
        # Initialize preference matrix
        for _, male in male_data.iterrows():
            male_id = male['id']
            self.preference_matrix[male_id] = {}
            
            # Extract preference ratings for each female
            for female_id in self.female_students:
                # Get rating column name
                rating_col = f'rating_{female_id}'
                if rating_col in male.index:
                    self.preference_matrix[male_id][female_id] = male[rating_col]
                else:
                    # If explicit rating not provided, use a neutral score
                    self.preference_matrix[male_id][female_id] = 5
        
        for _, female in female_data.iterrows():
            female_id = female['id']
            self.preference_matrix[female_id] = {}
            
            # Extract preference ratings for each male
            for male_id in self.male_students:
                # Get rating column name
                rating_col = f'rating_{male_id}'
                if rating_col in female.index:
                    self.preference_matrix[female_id][male_id] = female[rating_col]
                else:
                    # If explicit rating not provided, use a neutral score
                    self.preference_matrix[female_id][male_id] = 5
    
    def _create_feature_vectors(self):
        """Create feature vectors for each student using one-hot encoding for categorical variables
        and proper feature normalization."""
        # Define categorical features to encode
        categorical_features = ['preferred_activities', 'music_preference', 'personality']
        
        # Create encoders for each categorical feature
        for feature in categorical_features:
            unique_values = self.data[feature].unique()
            # Reshape to 2D array for fitting
            unique_values_2d = np.array(unique_values).reshape(-1, 1)
            encoder = OneHotEncoder(sparse_output=False)
            encoder.fit(unique_values_2d)
            self.encoders[feature] = encoder
        
        # Step 1: Create raw feature vectors for all students
        raw_feature_vectors = {}
        for _, student in self.data.iterrows():
            student_id = student['id']
            
            # Initialize feature vector with numeric features
            feature_vector = [
                float(student['height']),  # Store raw height for normalization later
            ]
            
            # Add one-hot encoded categorical features
            for feature in categorical_features:
                value = np.array([[student[feature]]])  # Convert to 2D array for transform
                encoded = self.encoders[feature].transform(value)
                feature_vector.extend(encoded.flatten())
            
            # Store the raw feature vector
            raw_feature_vectors[student_id] = np.array(feature_vector)
        
        # Step 2: Calculate normalization statistics for numeric features
        # Only normalize the first feature (height) - one-hot encoded features are already normalized
        heights = [vec[0] for vec in raw_feature_vectors.values()]
        height_min = min(heights)
        height_max = max(heights)
        height_range = height_max - height_min if height_max > height_min else 1.0
        
        # Step 3: Create normalized feature vectors
        for student_id, raw_vector in raw_feature_vectors.items():
            # Normalize numeric features (just height in this case)
            normalized_height = (raw_vector[0] - height_min) / height_range
            
            # Create a new vector with normalized height and keep one-hot encodings
            normalized_vector = np.copy(raw_vector)
            normalized_vector[0] = normalized_height
            
            # Store the normalized feature vector
            self.feature_vectors[student_id] = normalized_vector
            
        print(f"Feature vectors created with normalized height (range: {height_min:.1f}-{height_max:.1f})")
        
        # Step 4: Scale categorical features to have appropriate weight relative to numeric
        # Since one-hot encoding creates many binary features, we need to scale them appropriately
        # The number of categorical features is the sum of unique values for each categorical variable
        n_categorical_features = sum(len(self.encoders[feature].categories_[0]) for feature in categorical_features)
        
        # Calculate the average weighting factor for categorical features
        # This balances the influence of categorical vs. numeric features
        categorical_weight = 1.0 / n_categorical_features if n_categorical_features > 0 else 1.0
        
        # Apply weighting to each feature vector
        for student_id, vector in self.feature_vectors.items():
            # First feature is height (normalized 0-1) - leave it with weight 1.0
            # Apply weights to categorical features (the rest of the vector)
            weighted_vector = np.copy(vector)
            weighted_vector[1:] *= categorical_weight
            self.feature_vectors[student_id] = weighted_vector
            
        print(f"Categorical features weighted by factor: {categorical_weight:.4f}")
    
    def _calculate_euclidean_compatibility(self):
        """Calculate compatibility scores based on Euclidean distance between feature vectors."""
        # Calculate Euclidean distance between all pairs
        distances = {}
        for male_id in self.male_students:
            distances[male_id] = {}
            for female_id in self.female_students:
                if male_id in self.feature_vectors and female_id in self.feature_vectors:
                    # Calculate Euclidean distance
                    dist = np.linalg.norm(self.feature_vectors[male_id] - self.feature_vectors[female_id])
                    distances[male_id][female_id] = dist
        
        # Normalize distances to a [0, 10] scale (10 is closest/most compatible)
        all_distances = [d for m in distances for d in distances[m].values()]
        max_dist = max(all_distances) if all_distances else 1.0
        
        # Update preference matrix with distance-based compatibility
        for male_id in self.male_students:
            for female_id in self.female_students:
                if male_id in distances and female_id in distances[male_id]:
                    # Convert distance to similarity score (10 = perfect match)
                    normalized_dist = distances[male_id][female_id] / max_dist
                    similarity_score = 10 * (1 - normalized_dist)
                    
                    # Update preference matrix (use 40% from explicit ratings, 60% from feature similarity)
                    if male_id in self.preference_matrix and female_id in self.preference_matrix[male_id]:
                        explicit_score = self.preference_matrix[male_id][female_id]
                        self.preference_matrix[male_id][female_id] = 0.4 * explicit_score + 0.6 * similarity_score
                    
                    if female_id in self.preference_matrix and male_id in self.preference_matrix[female_id]:
                        explicit_score = self.preference_matrix[female_id][male_id]
                        self.preference_matrix[female_id][male_id] = 0.4 * explicit_score + 0.6 * similarity_score
    
    def save_matches(self, matches: List[Tuple[str, str]], output_file: str):
        """Save the final matches to a CSV file."""
        matched_data = []
        
        # Calculate standard deviation of Euclidean distances for the solution
        distances = []
        for male_id, female_id in matches:
            if male_id in self.feature_vectors and female_id in self.feature_vectors:
                dist = np.linalg.norm(self.feature_vectors[male_id] - self.feature_vectors[female_id])
                distances.append(dist)
        
        std_dev = np.std(distances) if distances else 0
        
        # Collect match data
        for male_id, female_id in matches:
            male_info = self.data[self.data['id'] == male_id].iloc[0]
            female_info = self.data[self.data['id'] == female_id].iloc[0]
            
            # Calculate Euclidean distance for this pair
            euclidean_distance = np.linalg.norm(
                self.feature_vectors[male_id] - self.feature_vectors[female_id]
            ) if male_id in self.feature_vectors and female_id in self.feature_vectors else None
            
            match_score = (self.preference_matrix[male_id][female_id] + 
                          self.preference_matrix[female_id][male_id]) / 2
            
            matched_data.append({
                'male_id': male_id,
                'male_name': male_info['name'],
                'female_id': female_id,
                'female_name': female_info['name'],
                'match_score': match_score,
                'euclidean_distance': euclidean_distance,
                'male_preference': self.preference_matrix[male_id][female_id],
                'female_preference': self.preference_matrix[female_id][male_id],
                'compatible_activities': male_info['preferred_activities'] == female_info['preferred_activities'],
                'compatible_music': male_info['music_preference'] == female_info['music_preference'],
                'height_difference': abs(male_info['height'] - female_info['height'])
            })
        
        # Create and save DataFrame
        matches_df = pd.DataFrame(matched_data)
        
        # Add solution metadata
        matches_df.attrs['std_dev_euclidean'] = std_dev
        
        # Save to CSV
        matches_df.to_csv(output_file, index=False)
        print(f"Matches saved to {output_file}")
        print(f"Standard deviation of Euclidean distances: {std_dev:.4f}")