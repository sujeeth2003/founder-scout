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

