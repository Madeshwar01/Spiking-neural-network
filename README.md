# Implementation of Neural Encoding Algorithms of Spiking Neural Networks

SNNs are inspired by the brain's functioning which closely mimic the behavior of biological neurons, particularly its energy-efficient and parallel processing capabilities.The study explores various encoding techniques such as Rate coding, Phase coding, Burst and Time-To-First-Spike (TTFS) coding using EEG signals’ dataset.

*Temporal coding strategies, such as TTFS (Time-To-First-Spike) coding, offer the fastest performance with lower latency and fewer operations compared to other encoding techniques. This makes them highly efficient.
*Rate coding, though stable, is less energy-efficient, while temporal coding and optimized hardware implementations, including FPGA-based designs, significantly reduce power consumption.
*Phase coding is represented by the relative timing or phase of spikes concerning a reference oscillatory pattern.
*Burst coding is a temporal coding strategy used in spiking neural networks (SNNs) where a neuron encodes information through a rapid sequence, or burst, of spikes.
<img width="588" alt="image" src="https://github.com/user-attachments/assets/75e35239-b1b7-44d6-a097-7aed03112814">

EEG Signal Preprocessing and BRAM Storage:
EEG dataset is preprocessed and stored in a 32-bit wide, 784-depth BRAM IP core created in Vivado, enabling efficient storage and access.


BRAM Data Retrieval:
A memory_read module with a 10-bit address counter sequentially accesses 32-bit data from BRAM, incrementing each clock cycle and resetting at 784 depth.


SNN Encoding:
MSBs of BRAM data are fed into SNN encoding modules (rate coding, burst coding, phase coding, TTFS coding) to generate spike trains for neural network processing.

Flowchart for SNN classifier in Python:

<img width="647" alt="image" src="https://github.com/user-attachments/assets/47c9e701-843c-4134-9f36-88314a7a95d3">


Our project investigates the implementation of neural encoding algorithms in Spiking Neural Networks (SNNs) using the EEG signals’ dataset, focusing on optimizing performance.
The project lays the groundwork for future research, particularly in extending these encoding strategies to practical applications such as medical diagnosis. 

