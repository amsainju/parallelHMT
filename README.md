# Parallelizing Geographical Hidden Markov Tree for Flood Mapping

This is a parallel program to speed up the parameter learning of HMT [1,2] and HMT+ [2], to be run in a multi-core machine

Since we use OpenMP, this is only for running on top of Linux

### How to Configure

1. Open Da_Merged.cpp
2. If you are using HMT, make sure Line 7 "#define INCLUDE_M" is commented out; while if you are using HMT+ make sure Line 7 "#define INCLUDE_M" is enabled
3. If you are running actual program to count the running time, comment out Line 5 "#define DEBUG" as it will take additional time
4. Line 16 "#define Dim 3" is for input data with 3-dimensional feature vectors (e.g., RGB); change the dimension number according to your feature vector length

### How to Compile

1. Change your current directory to be the directory containing Da_Merged.cpp
2. In the console, compile with: **g++ -std=c++11 -O2 -g -fopenmp Da_Merged.cpp -o run**
3. The "run" program is generated for your use, and you can rename it to what you like such as "hmt" or "hmtplus"

### How to Run

1. We have 4 data folders for you to test out, 2 for HMT and 2 for HMT+. Toy data should use "#define Dim 1" while real data should use "#define Dim 3".
2. To run the compiled HMT program "run" over real data (the large one), use: ./run RealData_HMT/configHMT.txt

### References

[1] Miao Xie, Zhe Jiang, Arpan Man Sainju, "Geographical Hidden Markov Tree for Flood Extent Mapping," KDD 2018: 2545-2554

[2] Zhe Jiang, Miao Xie and Arpan Man Sainju, "Geographical Hidden Markov Tree," in IEEE Transactions on Knowledge and Data Engineering