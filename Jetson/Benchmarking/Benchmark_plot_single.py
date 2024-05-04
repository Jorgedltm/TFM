import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    
    df = pd.read_csv('CSV_files/Benchmark_single_RIR_jn.csv',sep=';',decimal=',')
    libs = df['Library'].values
    lengths = df['Length_RIR'].values
    mean = df['Mean'].values
    std = df['Std_dev'].values
    unique_libs = np.unique(libs)
    lengths_u = np.unique(lengths)
    index = []
    for length in lengths_u:
        index = np.append(index,np.where(lengths == length)[0])
    libs = libs[index.astype(int)]
    mean = mean[index.astype(int)]
    std = std[index.astype(int)]
    
    # Plot values
    plt.figure(figsize=(10,5))
    for lib in unique_libs:
        plt.errorbar(lengths_u,mean[np.where(libs == lib)[0]],std[np.where(libs == lib)[0]],fmt='-o',linewidth=2, capsize=4)
    plt.legend(unique_libs)
    plt.yscale('log')
    plt.xlabel("Length of RIR (s)")
    plt.ylabel("Execution time (ms)")
    plt.title("Execution time of STFT vs length of RIR on Jetson Nano")
    plt.savefig("Figures/Benchmarking_single_RIR_jn_log.png")
    plt.show()

if __name__ == "__main__":
  main()
