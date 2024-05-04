import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    
    df = pd.read_csv('CSV_files/Benchmark_multiple_RIRs.csv',sep=';',decimal=',')
    libs = df['Library'].values
    nreps = df['Num_reps'].values
    mean = df['Mean'].values
    std = df['Std_dev'].values
    unique_libs = np.unique(libs)
    nreps_u = np.unique(nreps)
    index = []
    for nrep in nreps_u:
        index = np.append(index,np.where(nreps == nrep)[0])
    libs = libs[index.astype(int)]
    mean = mean[index.astype(int)]
    std = std[index.astype(int)]
    
    # Plot values
    plt.figure(figsize=(10,5))
    for lib in unique_libs:
        plt.errorbar(nreps_u,mean[np.where(libs == lib)[0]],std[np.where(libs == lib)[0]],fmt='-o',linewidth=2, capsize=4)
    plt.legend(unique_libs)
    plt.xlabel("Number of RIRs")
    plt.ylabel("Execution time (ms)")
    plt.title("Execution time of STFT vs number of stacked RIRs on Google Computer")
    plt.savefig("Figures/Benchmarking_multiple_RIRs_Google_computer.png")
    plt.show()

if __name__ == "__main__":
  main()
