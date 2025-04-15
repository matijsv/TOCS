import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define an exponential decay function
def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def analyze_spikes(data):
    time_intervals = np.diff(data)

    hist, bin_edges = np.histogram(time_intervals, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    params, _ = curve_fit(exponential_decay, bin_centers, hist, p0=(1, 1))

    decay_rate = params[1]

    plt.hist(time_intervals, bins=50, density=True, alpha=0.6, edgecolor='black', label='Histogram')
    plt.plot(bin_centers, exponential_decay(bin_centers, *params), color='red', label='Exponential Fit')
    plt.xlabel('Time Interval (s)')
    plt.ylabel('Density')
    plt.title('Exponential Fit to Time Interval Distribution')
    plt.legend()
    plt.show()
    
    return time_intervals, decay_rate

def spiking_model(n_spikes, lam, tau0, seed=None):
    """
    Simulates a delayed Poisson process until a fixed number of spikes is reached.

    Parameters:
    -----------
    n_spikes : int
        Number of spikes to simulate
    lam : float
        Decay rate
    tau0 : float
        Refractory delay (seconds)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    spike_times : list of float
        Times at which spikes occur
    """
    if seed is not None:
        np.random.seed(seed)

    spike_times = []
    t = 0.0

    for _ in range(n_spikes):
        e = np.random.exponential(scale=1/lam)
        t += tau0 + e # simply shifts the next 
        spike_times.append(t)

    return spike_times

def avg_spiking_rate(data):
    n_spikes = len(data)
    print("   Datapoints:", n_spikes)
    return n_spikes / data[-1]

if __name__ == "__main__":
    # region 1.2 & 1.2
    actual_spike_times = np.loadtxt('./data/Data_neuron.txt')
    
    actual_time_intervals, fitted_decay_rate = analyze_spikes(actual_spike_times)
    
    tau_0 = np.min(actual_time_intervals)
    print("\n", "Actual Data Analysis results:")
    print("   Minimum tau recorded (tau_0):", tau_0)
    print("   Decay rate fitted (lambda):", fitted_decay_rate)

    # region 1.3 & 1.4
    simulated_spike_times = spiking_model(1000, fitted_decay_rate, tau_0)
    simulated_time_intervals, simulated_decay_rate = analyze_spikes(simulated_spike_times)
    
    retreived_tau_0 = np.min(simulated_time_intervals)
    print("\n", "Simulated Data Analysis Results:")
    print("   retreived tau_0:", retreived_tau_0)
    print("   retreived decay rate:", simulated_decay_rate)
    
    # region 1.5
    print("\n", "Average Spiking Rates:")
    print("   Actual:", avg_spiking_rate(actual_spike_times))
    print("   Simulated:", avg_spiking_rate(simulated_spike_times))