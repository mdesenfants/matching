#!/usr/bin/env python3
import argparse
import time
import matplotlib.pyplot as plt
from typing import List, Tuple

from data_loader import DataLoader
from annealing import SimulatedAnnealing

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Match prom dates using simulated annealing')
    parser.add_argument('--input', required=True, help='Input CSV file with student data')
    parser.add_argument('--output', default='matches.csv', help='Output CSV file for matches')
    parser.add_argument('--iterations', type=int, default=10000, help='Maximum iterations for annealing')
    parser.add_argument('--temp', type=float, default=100.0, help='Initial temperature')
    parser.add_argument('--cooling-rate', type=float, default=0.995, help='Cooling rate')
    parser.add_argument('--preference-weight', type=float, default=0.6, 
                        help='Weight for preference scores (0-1)')
    parser.add_argument('--distance-weight', type=float, default=0.4, 
                        help='Weight for distance objective (0-1)')
    parser.add_argument('--objective', type=str, default='minimize_sum', 
                        choices=['minimize_sum', 'minimize_std_dev'],
                        help='Objective for distance optimization')
    parser.add_argument('--visualize', action='store_true', help='Visualize the annealing process')
    
    args = parser.parse_args()
    
    # Check weight args
    weight_sum = args.preference_weight + args.distance_weight
    if abs(weight_sum - 1.0) > 0.0001:
        print(f"Warning: Weights should sum to 1.0, got {weight_sum}. Normalizing.")
        norm_factor = 1.0 / weight_sum
        args.preference_weight *= norm_factor
        args.distance_weight *= norm_factor
    
    # Load and prepare data
    print(f"Loading data from {args.input}...")
    loader = DataLoader(args.input)
    male_students, female_students, preference_matrix = loader.prepare_data()
    print(f"Loaded {len(male_students)} male and {len(female_students)} female students.")
    
    # Set up visualization if requested
    if args.visualize:
        plt.figure(figsize=(12, 8))
        plt.title('Simulated Annealing Progress')
        
        # These will store data for plotting
        iterations = []
        scores = []
        best_scores = []
        temperatures = []
        std_devs = []
        
        # Define callback for visualization
        def callback(iteration, temp, current_score, best_score, best_solution):
            # Calculate current standard deviation
            distances = []
            for male_id, female_id in best_solution:
                if 'dummy' in male_id or 'dummy' in female_id:
                    continue
                if (male_id in loader.feature_vectors and 
                    female_id in loader.feature_vectors):
                    dist = np.linalg.norm(
                        loader.feature_vectors[male_id] - loader.feature_vectors[female_id]
                    )
                    distances.append(dist)
            
            current_std_dev = np.std(distances) if len(distances) > 1 else 0
            
            iterations.append(iteration)
            scores.append(current_score)
            best_scores.append(best_score)
            temperatures.append(temp)
            std_devs.append(current_std_dev)
            
            # Update plot every 100 iterations to avoid slowing down the process
            if iteration % 100 == 0:
                plt.clf()
                plt.subplot(2, 2, 1)
                plt.plot(iterations, scores, 'b-', alpha=0.3, label='Current Score')
                plt.plot(iterations, best_scores, 'r-', label='Best Score')
                plt.title('Score Evolution')
                plt.xlabel('Iteration')
                plt.ylabel('Score')
                plt.legend()
                plt.grid(True)
                
                # Calculate and track min/max/avg scores to display range
                if iteration % 100 == 0:
                    plt.text(0.05, 0.05, 
                             f"Min: {min(best_scores):.2f}\nMax: {max(best_scores):.2f}\nRange: {max(best_scores)-min(best_scores):.2f}", 
                             transform=plt.gca().transAxes)
                
                plt.subplot(2, 2, 2)
                plt.plot(iterations, temperatures, 'g-')
                plt.title('Temperature Decay')
                plt.xlabel('Iteration')
                plt.ylabel('Temperature')
                plt.grid(True)
                
                plt.subplot(2, 2, 3)
                plt.plot(iterations, std_devs, 'm-')
                plt.title('Standard Deviation of Euclidean Distances')
                plt.xlabel('Iteration')
                plt.ylabel('Std Dev')
                plt.grid(True)
                
                # Track min/max/avg std dev
                plt.text(0.05, 0.05, 
                         f"Min: {min(std_devs):.4f}\nMax: {max(std_devs):.4f}\nRange: {max(std_devs)-min(std_devs):.4f}", 
                         transform=plt.gca().transAxes)
                
                # Plot current best matches as a matrix
                if iteration % 1000 == 0 and len(best_solution) > 0:
                    plt.subplot(2, 2, 4)
                    
                    # Create a compatibility matrix for visualization
                    real_matches = [(m, f) for m, f in best_solution if 'dummy' not in m and 'dummy' not in f]
                    if real_matches:
                        match_dict = dict(real_matches)
                        compatibility = []
                        for m_id in male_students:
                            row = []
                            for f_id in female_students:
                                # Highlight actual matches
                                if f_id == match_dict.get(m_id, None):
                                    row.append(1)  # Paired
                                else:
                                    row.append(0)  # Not paired
                            compatibility.append(row)
                        
                        plt.imshow(compatibility, cmap='viridis', interpolation='nearest')
                        plt.title(f'Current Best Matches (iter {iteration})')
                        plt.xlabel('Female Students')
                        plt.ylabel('Male Students')
                        plt.colorbar(ticks=[0, 1], label='Paired')
                
                plt.tight_layout()
                plt.pause(0.001)
    else:
        callback = None
    
    # Run simulated annealing
    print("Running simulated annealing...")
    start_time = time.time()
    
    annealer = SimulatedAnnealing(
        male_students=male_students,
        female_students=female_students,
        preference_matrix=preference_matrix,
        feature_vectors=loader.feature_vectors,
        initial_temp=args.temp,
        cooling_rate=args.cooling_rate,
        preference_weight=args.preference_weight,
        distance_weight=args.distance_weight,
        objective=args.objective
    )
    
    best_matches = annealer.run(max_iterations=args.iterations, callback=callback)
    
    end_time = time.time()
    print(f"Annealing completed in {end_time - start_time:.2f} seconds.")
    
    # Save matches to output file
    loader.save_matches(best_matches, args.output)
    
    # Calculate and display statistics for the matches
    total_preference = 0
    distances = []
    
    # Display final matches
    print("\nFinal Matches:")
    for male_id, female_id in best_matches:
        male_data = loader.data[loader.data['id'] == male_id].iloc[0]
        female_data = loader.data[loader.data['id'] == female_id].iloc[0]
        
        m_score = preference_matrix[male_id][female_id]
        f_score = preference_matrix[female_id][male_id]
        avg_score = (m_score + f_score) / 2
        total_preference += avg_score
        
        # Calculate Euclidean distance
        if (male_id in loader.feature_vectors and female_id in loader.feature_vectors):
            dist = np.linalg.norm(
                loader.feature_vectors[male_id] - loader.feature_vectors[female_id]
            )
            distances.append(dist)
            distance_str = f", Euclidean dist: {dist:.3f}"
        else:
            distance_str = ""
        
        print(f"{male_data['name']} ({male_id}) & {female_data['name']} ({female_id}) "
              f"- Score: {avg_score:.1f} ({m_score:.1f}/{f_score:.1f}){distance_str}")
    
    # Calculate statistics
    avg_preference = total_preference / len(best_matches) if best_matches else 0
    total_distance = sum(distances) if distances else 0
    avg_distance = total_distance / len(distances) if distances else 0
    std_dev = np.std(distances) if len(distances) > 1 else 0
    
    print(f"\nSummary Statistics:")
    print(f"Average preference score: {avg_preference:.2f}")
    print(f"Total Euclidean distance: {total_distance:.4f}")
    print(f"Average Euclidean distance: {avg_distance:.4f}")
    print(f"Standard deviation of Euclidean distances: {std_dev:.4f}")
    
    # Show final plot if visualizing
    if args.visualize:
        plt.show()

if __name__ == "__main__":
    import numpy as np  # Import here for callback function
    main()