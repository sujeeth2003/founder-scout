import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

num_rows = 100

names = [
    "Alice Chen", "Bob Smith", "Charlie Garcia", "Diana Prince", "Ethan Hunt",
    "Fiona Gallagher", "George Costanza", "Hannah Abbott", "Ian Wright", "Julia Roberts",
    "Kevin Hart", "Lara Croft", "Miles Morales", "Nina Simone", "Oscar Isaac",
    "Peter Parker", "Quinn Fabray", "Riley Reid", "Sam Winchester", "Tony Stark",
    "Bruce Wayne", "Clark Kent", "Wanda Maximoff", "Steve Rogers", "Natasha Romanoff",
    "Barry Allen", "Arthur Curry", "Victor Stone", "Hal Jordan", "Oliver Queen",
    "Dinah Lance", "John Constantine", "Zatanna Zatara", "Billy Batson", "Shazam",
    "Carol Danvers", "T'Challa", "Scott Lang", "Hope van Dyne", "Stephen Strange",
    "Peter Quill", "Gamora", "Drax", "Rocket Raccoon", "Groot", "Mantice",
    "Nebula", "Loki", "Thor Odinson", "Jane Foster", "Peggy Carter", "Bucky Barnes",
    "Sam Wilson", "James Rhodes", "Vision", "Nick Fury", "Maria Hill", "Phil Coulson",
    "Melinda May", "Daisy Johnson", "Grant Ward", "Jemma Simmons", "Leo Fitz",
    "Elena Rodriguez", "Alphonso Mackenzie", "Lance Hunter", "Bobbi Morse",
    "Luke Cage", "Jessica Jones", "Matt Murdock", "Danny Rand", "Frank Castle",
    "Claire Temple", "Colleen Wing", "Misty Knight", "Wilson Fisk", "Vanessa Marianna",
    "Karen Page", "Foggy Nelson", "Ben Urich", "Leland Owlsley", "Turk Barrett",
    "Brett Mahoney", "Marci Stahl", "Blake Tower", "Jeri Hogarth", "Malcolm Ducasse",
    "Trish Walker", "Will Simpson", "Hope Shlottman", "Kilgrave", "Reva Connors",
    "Cottonmouth", "Mariah Dillard", "Shades Alvarez", "Diamondback", "Pop"
]

founder_names = [names[i % len(names)] for i in range(num_rows)]

# 1. Base Core Features
prior_exits = np.random.choice([0, 1, 2, 3], size=num_rows, p=[0.4, 0.3, 0.2, 0.1])
has_prior_exit = (prior_exits > 0).astype(int)
team_size = np.random.randint(2, 26, size=num_rows)
domain_match = np.round(np.random.uniform(0.0, 1.0, size=num_rows), 2)
raised_seed = np.random.choice([1, 0], size=num_rows, p=[0.7, 0.3])

# 2. Technical & GitHub Features
github_stars_log = np.round(np.random.gamma(2, 2, size=num_rows), 2)
founder_follower_log = np.round(github_stars_log * 0.6 + np.random.normal(0, 0.5, size=num_rows), 2)
has_ml_repos = np.random.choice([1, 0], size=num_rows, p=[0.4, 0.6])
num_languages = np.random.poisson(lam=4, size=num_rows).clip(1, 15)

# 3. New Calculated & Metadata Columns
tech_depth_score = np.round((github_stars_log * 0.4) + (has_ml_repos * 2) + (num_languages * 0.2), 2)
fund_thesis_overlap = np.round(np.random.beta(5, 2, size=num_rows), 2)
account_age_years = np.random.randint(1, 15, size=num_rows)

# 4. Success Logic (Weighted combination of features)
success_prob = (
    (prior_exits * 0.15) + 
    (raised_seed * 0.2) + 
    (domain_match * 0.2) +
    (tech_depth_score * 0.05) +
    (has_ml_repos * 0.1)
)
success_prob = np.clip(success_prob / success_prob.max(), 0, 1)
series_a_success = (np.random.rand(num_rows) < success_prob).astype(int)

