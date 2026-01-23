"""
model.py
--------
Trains a Gradient Boosting classifier to predict Series A success.
Saves model artifact to outputs/model.pkl.

Features:
  - Founder pedigree (exits, education)
  - Team composition
  - Funding signals
  - Technical traction (GitHub proxy / real GitHub features if fetched)
  - Domain overlap with fund thesis
"""

