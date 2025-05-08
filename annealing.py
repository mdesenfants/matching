#!/usr/bin/env python3
import random
import math
import numpy as np
from typing import List, Dict, Tuple, Callable

class SimulatedAnnealing:
    def __init__(self, 
                 male_students: List[str], 
                 female_students: List[str],
                 preference_matrix: Dict[str, Dict[str, float]],
                 feature_vectors: Dict[str, np.ndarray] = None,
                 initial_temp: float = 100.0,
                 cooling_rate: float = 0.995,
                 min_temp: float = 0.01,
                 preference_weight: float = 0.6,
                 distance_weight: float = 0.4,
                 objective: str = "minimize_sum"):
        """Initialize the simulated annealing optimizer.
        
        Args:
            male_students: List of male student IDs
            female_students: List of female student IDs
            preference_matrix: Dictionary mapping student IDs to their preferences
            feature_vectors: Dictionary mapping student IDs to feature vectors for Euclidean distance
            initial_temp: Starting temperature for simulated annealing
            cooling_rate: Rate at which temperature decreases each iteration
            min_temp: Minimum temperature at which to stop the annealing process
            preference_weight: Weight given to preference scores (0-1)
            distance_weight: Weight given to distance objective (0-1)
            objective: Optimization objective for distances ("minimize_sum" or "minimize_std_dev")
        """
        self.male_students = male_students.copy()
        self.female_students = female_students.copy()
        self.preference_matrix = preference_matrix
        self.feature_vectors = feature_vectors or {}
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.preference_weight = preference_weight
        self.distance_weight = distance_weight
        self.objective = objective
        
        # Make sure we have equal numbers of male and female students
        # If not, we'll add dummy students
        self._balance_student_numbers()
    
    def _balance_student_numbers(self):
        """Ensure equal numbers of male and female students by adding dummy entries if needed."""
        male_count = len(self.male_students)
        female_count = len(self.female_students)
        
        # If already balanced, do nothing
        if male_count == female_count:
            return
        
        # Add dummy students to the smaller group
        if male_count < female_count:
            dummy_count = female_count - male_count
            for i in range(dummy_count):
                dummy_id = f"M_dummy_{i+1}"
                self.male_students.append(dummy_id)
                
                # Add neutral preferences for this dummy student
                self.preference_matrix[dummy_id] = {}
                for female_id in self.female_students:
                    self.preference_matrix[dummy_id][female_id] = 1  # Low preference
                    
                    # Also add preference from females to this dummy
                    if female_id in self.preference_matrix:
                        self.preference_matrix[female_id][dummy_id] = 1  # Low preference
                    else:
                        self.preference_matrix[female_id] = {dummy_id: 1}
                
                # Add a zero feature vector for Euclidean distance
                if self.feature_vectors:
                    # Get feature vector length from an existing vector
                    example_id = next(iter(self.feature_vectors))
                    vec_len = len(self.feature_vectors[example_id])
                    self.feature_vectors[dummy_id] = np.zeros(vec_len)
                
        elif female_count < male_count:
            dummy_count = male_count - female_count
            for i in range(dummy_count):
                dummy_id = f"F_dummy_{i+1}"
                self.female_students.append(dummy_id)
                
                # Add neutral preferences for this dummy student
                self.preference_matrix[dummy_id] = {}
                for male_id in self.male_students:
                    self.preference_matrix[dummy_id][male_id] = 1  # Low preference
                    
                    # Also add preference from males to this dummy
                    if male_id in self.preference_matrix:
                        self.preference_matrix[male_id][dummy_id] = 1  # Low preference
                    else:
                        self.preference_matrix[male_id] = {dummy_id: 1}
                
                # Add a zero feature vector for Euclidean distance
                if self.feature_vectors:
                    # Get feature vector length from an existing vector
                    example_id = next(iter(self.feature_vectors))
                    vec_len = len(self.feature_vectors[example_id])
                    self.feature_vectors[dummy_id] = np.zeros(vec_len)
    
    def _generate_initial_solution(self) -> List[Tuple[str, str]]:
        """Generate an initial random matching of male and female students."""
        # Randomly shuffle both lists to ensure unique pairings
        male_shuffled = self.male_students.copy()
        female_shuffled = self.female_students.copy()
        random.shuffle(male_shuffled)
        random.shuffle(female_shuffled)
        
        # Create pairs by matching students at the same index
        return [(male_shuffled[i], female_shuffled[i]) 
                for i in range(len(male_shuffled))]
    
    def _evaluate_solution(self, solution: List[Tuple[str, str]]) -> float:
        """Calculate the combined score for a given matching solution.
        
        Combines:
        1. Preference satisfaction (higher is better)
        2. Distance objective (depending on self.objective):
           - "minimize_sum": Sum of Euclidean distances (lower is better)
           - "minimize_std_dev": Standard deviation of distances (lower is better)
        """
        # Calculate preference score (higher is better)
        preference_score = 0
        real_pair_count = 0
        
        # Calculate Euclidean distances
        distances = []
        
        for male_id, female_id in solution:
            # Skip dummy students in the evaluation
            if 'dummy' in male_id or 'dummy' in female_id:
                continue
            
            real_pair_count += 1
                
            # Add male's preference for female
            if male_id in self.preference_matrix and female_id in self.preference_matrix[male_id]:
                preference_score += self.preference_matrix[male_id][female_id]
            
            # Add female's preference for male
            if female_id in self.preference_matrix and male_id in self.preference_matrix[female_id]:
                preference_score += self.preference_matrix[female_id][male_id]
            
            # Calculate Euclidean distance
            if (self.feature_vectors and male_id in self.feature_vectors and 
                female_id in self.feature_vectors):
                dist = np.linalg.norm(self.feature_vectors[male_id] - self.feature_vectors[female_id])
                distances.append(dist)
        
        # Normalize preference score by number of real pairs
        if real_pair_count > 0:
            preference_score /= (real_pair_count * 2)  # Divide by pairs * 2 (male+female preference)
        else:
            preference_score = 0
        
        # Calculate distance objective score based on selected objective
        if self.objective == "minimize_sum":
            # Sum of distances (lower is better)
            total_distance = sum(distances) if distances else 0
            
            # Normalize to a 0-10 scale where 10 is best (lowest distance)
            # The constant 5.0 is a calibration factor that may need adjustment based on data
            distance_score = 10 * math.exp(-total_distance / (real_pair_count * 0.5))
            
        else:  # minimize_std_dev
            # Calculate standard deviation (lower is better)
            std_dev = np.std(distances) if len(distances) > 1 else 0
            
            # Calculate coefficient of variation (CV) to have a scale-independent measure
            mean_dist = np.mean(distances) if distances else 1.0
            cv = std_dev / mean_dist if mean_dist > 0 else 0
            
            # Normalize CV to a 0-10 scale (10 is best = lowest CV)
            distance_score = 10 * math.exp(-5 * cv)
        
        # Combine the scores with weights
        # For preference score, higher is better
        # For distance score, higher is better (since we normalized it)
        combined_score = (self.preference_weight * preference_score + 
                         self.distance_weight * distance_score)
        
        return combined_score
    
    def _get_neighbor_solution(self, current_solution: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Generate a neighbor solution by swapping female students between two male students.
        
        This maintains the one-to-one matching constraint by always swapping pairs.
        """
        # Copy the current solution
        neighbor = current_solution.copy()
        
        # Randomly select two different indices
        idx1, idx2 = random.sample(range(len(neighbor)), 2)
        
        # Save the original pairs
        male1, female1 = neighbor[idx1]
        male2, female2 = neighbor[idx2]
        
        # Swap the female students (keeping the male students in place)
        neighbor[idx1] = (male1, female2)
        neighbor[idx2] = (male2, female1)
        
        return neighbor
    
    def run(self, max_iterations: int = 10000, callback: Callable = None) -> List[Tuple[str, str]]:
        """Run the simulated annealing algorithm to find optimal matches.
        
        Args:
            max_iterations: Maximum number of iterations to perform
            callback: Optional callback function called after each iteration with
                      (iteration, temperature, current_score, best_score, best_solution)
        
        Returns:
            List of tuples representing the best matching found
        """
        # Generate initial solution
        current_solution = self._generate_initial_solution()
        current_score = self._evaluate_solution(current_solution)
        
        # Keep track of the best solution
        best_solution = current_solution.copy()
        best_score = current_score
        
        # Initial temperature
        temperature = self.initial_temp
        
        # Main annealing loop
        iteration = 0
        while temperature > self.min_temp and iteration < max_iterations:
            # Generate neighbor solution
            neighbor_solution = self._get_neighbor_solution(current_solution)
            neighbor_score = self._evaluate_solution(neighbor_solution)
            
            # Calculate score difference
            score_diff = neighbor_score - current_score
            
            # Decide whether to accept the neighbor solution
            if score_diff > 0:  # Better solution, always accept
                current_solution = neighbor_solution
                current_score = neighbor_score
            else:  # Worse solution, accept with some probability
                # Calculate acceptance probability
                accept_prob = math.exp(score_diff / temperature)
                
                # Accept with calculated probability
                if random.random() < accept_prob:
                    current_solution = neighbor_solution
                    current_score = neighbor_score
            
            # Update best solution if current solution is better
            if current_score > best_score:
                best_solution = current_solution.copy()
                best_score = current_score
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
            
            # Call callback if provided
            if callback and iteration % 100 == 0:  # Call every 100 iterations to reduce overhead
                callback(iteration, temperature, current_score, best_score, best_solution)
        
        # Filter out dummy students from the final solution
        final_solution = [(male, female) for male, female in best_solution 
                          if 'dummy' not in male and 'dummy' not in female]
        
        return final_solution