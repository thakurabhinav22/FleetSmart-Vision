import numpy as np
import csv

CLASSES = ['active', 'sleeping', 'yawning']
CSV_FILE = 'dataset.csv'

print("Generating 15,000 professional NORMALIZED samples...")
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    for _ in range(5000):
        # 1. ACTIVE (Values should hover around 1.0 of baseline)
        writer.writerow([0, np.random.normal(1.0, 0.05), np.random.normal(1.0, 0.05), np.random.normal(1.0, 0.2)])
        # 2. SLEEPING (Eyes drop significantly below 0.6 of baseline)
        writer.writerow([1, np.random.normal(0.2, 0.15), np.random.normal(0.2, 0.15), np.random.normal(1.0, 0.2)])
        # 3. YAWNING (Mouth expands to > 2.5x of baseline)
        writer.writerow([2, np.random.normal(0.8, 0.15), np.random.normal(0.8, 0.15), np.random.normal(4.0, 0.8)])

print("Done! Ready to train universal model.")
