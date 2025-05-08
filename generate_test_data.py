#!/usr/bin/env python3
import csv
import random
import argparse
from typing import List, Dict

def generate_students(num_male: int = 10, num_female: int = 10) -> List[Dict]:
    """Generate a list of student data with more diverse preferences."""
    
    # Create student IDs and basic info
    students = []
    male_names = ["Alex", "Ben", "Carlos", "David", "Ethan", "Frank", "George", "Harry", "Ian", "Jack"]
    female_names = ["Amy", "Beth", "Chloe", "Diana", "Emma", "Fiona", "Grace", "Hannah", "Isla", "Jen"]
    
    # Generate preference categories with more options for diversity
    activities = ["Dancing", "Taking photos", "Talking", "Eating", "Group activities", 
                  "Games", "Singing", "Sports", "Arts", "Movies"]
    music_genres = ["Pop", "Hip hop", "Rock", "EDM", "Classical", 
                    "Country", "Jazz", "R&B", "Metal", "Indie"]
    personality_traits = ["Outgoing", "Reserved", "Funny", "Serious", "Creative", 
                         "Analytical", "Adventurous", "Calm", "Energetic", "Thoughtful"]
    
    # Generate male students
    for i in range(num_male):
        student = {
            "id": f"M{i+1}",
            "name": male_names[i % len(male_names)],
            "gender": "Male",
            "height": random.randint(165, 190),  # Height in cm
            "preferred_activities": random.choice(activities),
            "music_preference": random.choice(music_genres),
            "personality": random.choice(personality_traits)
        }
        
        # Generate highly variable preference ratings for opposite gender
        for j in range(num_female):
            # Creating significant variations in preferences
            student[f"rating_F{j+1}"] = random.randint(1, 10)
            
        students.append(student)
    
    # Generate female students
    for i in range(num_female):
        student = {
            "id": f"F{i+1}",
            "name": female_names[i % len(female_names)],
            "gender": "Female",
            "height": random.randint(155, 180),  # Height in cm
            "preferred_activities": random.choice(activities),
            "music_preference": random.choice(music_genres),
            "personality": random.choice(personality_traits)
        }
        
        # Generate highly variable preference ratings for opposite gender
        for j in range(num_male):
            # Creating significant variations in preferences  
            student[f"rating_M{j+1}"] = random.randint(1, 10)
            
        students.append(student)
    
    # Create some clusters of compatibility to make the problem more interesting
    # Group 1: Students who like similar activities
    activity_groups = {}
    for activity in activities:
        activity_groups[activity] = []
    
    for student in students:
        activity = student["preferred_activities"]
        activity_groups[activity].append(student["id"])
    
    # Group 2: Students with compatible personalities
    compatible_personalities = {
        "Outgoing": "Reserved",
        "Reserved": "Outgoing",
        "Funny": "Serious",
        "Serious": "Funny",
        "Creative": "Analytical",
        "Analytical": "Creative",
        "Adventurous": "Calm",
        "Calm": "Adventurous",
        "Energetic": "Thoughtful",
        "Thoughtful": "Energetic"
    }
    
    # Increase ratings for students with complementary personalities
    for student in students:
        personality = student["personality"]
        complementary = compatible_personalities.get(personality)
        
        if complementary:
            for other in students:
                if other["gender"] != student["gender"] and other["personality"] == complementary:
                    # Boost the rating
                    rating_key = f"rating_{other['id']}"
                    if rating_key in student:
                        student[rating_key] = min(10, student[rating_key] + random.randint(1, 3))
    
    # Group 3: Students with similar music preferences
    for student in students:
        music = student["music_preference"]
        for other in students:
            if other["gender"] != student["gender"] and other["music_preference"] == music:
                rating_key = f"rating_{other['id']}"
                if rating_key in student:
                    student[rating_key] = min(10, student[rating_key] + random.randint(1, 2))
    
    return students

def save_to_csv(students: List[Dict], filename: str) -> None:
    """Save the student data to a CSV file."""
    if not students:
        return
    
    # Get all unique keys from all students to form the header
    all_keys = set()
    for student in students:
        all_keys.update(student.keys())
    
    fieldnames = ["id", "name", "gender", "height", "preferred_activities", 
                  "music_preference", "personality"]
    
    # Add rating fields in a sorted order
    rating_fields = [key for key in all_keys if key.startswith("rating_")]
    rating_fields.sort()
    fieldnames.extend(rating_fields)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(students)
    
    print(f"Generated data saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate test data for prom matcher')
    parser.add_argument('--males', type=int, default=10, help='Number of male students')
    parser.add_argument('--females', type=int, default=10, help='Number of female students')
    parser.add_argument('--output', default='test_data.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    students = generate_students(args.males, args.females)
    save_to_csv(students, args.output)

if __name__ == "__main__":
    main()