import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

data = './data/US_SupremeCourt_n9_N895.txt'
with open(data, 'r') as file:
    data = file.readlines()
data = [[-1 if char == '0' else 1 for char in line.strip()] for line in data]

num_spin_variables = len(data[0])
two_power_n = 2 ** num_spin_variables

print(f"Number of spin variables: {num_spin_variables}")
print(f"Number of state possible (2^n): {two_power_n}")

print(f"Total number of datapoints: {len(data[0])*len(data)}")

unique_subarrays = np.unique(data, axis = 0)
print(f"Number of unique subarrays: {len(unique_subarrays)}")

mean_votes = np.mean(data, axis = 0)

ordered_mean_votes = np.sort(mean_votes)
plt.scatter(range(len(ordered_mean_votes)), ordered_mean_votes, color='black', s=10)
plt.show()

