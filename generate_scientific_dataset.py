import numpy as np
import csv

CLASSES = ['active', 'sleeping', 'yawning']
CSV_FILE = 'dataset.csv'

# Generate 5000 samples per class based on standard EAR/MAR research values
# Active: EAR ~ N(0.30, 0.04), MAR ~ N(0.05, 0.02)
# Sleeping: EAR ~ N(0.08, 0.03), MAR ~ N(0.05, 0.02)
# Yawning: EAR ~ N(0.25, 0.08), MAR ~ N(0.70, 0.15)

with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    
    # Active
    for _ in range(5000):
        l_ear = max(0.20, min(0.45, np.random.normal(0.32, 0.04)))
        r_ear = max(0.20, min(0.45, np.random.normal(0.32, 0.04)))
        mar = max(0.0, min(0.25, np.random.normal(0.05, 0.03)))
        writer.writerow([0, l_ear, r_ear, mar])
        
    # Sleeping
    for _ in range(5000):
        l_ear = max(0.0, min(0.16, np.random.normal(0.08, 0.03)))
        r_ear = max(0.0, min(0.16, np.random.normal(0.08, 0.03)))
        mar = max(0.0, min(0.25, np.random.normal(0.05, 0.03)))
        writer.writerow([1, l_ear, r_ear, mar])
        
    # Yawning
    for _ in range(5000):
        l_ear = max(0.10, min(0.45, np.random.normal(0.25, 0.08)))
        r_ear = max(0.10, min(0.45, np.random.normal(0.25, 0.08)))
        mar = max(0.35, min(1.5, np.random.normal(0.75, 0.15)))
        writer.writerow([2, l_ear, r_ear, mar])

print("Generated 15,000 samples in dataset.csv")
