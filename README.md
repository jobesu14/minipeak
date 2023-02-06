# Minipeak #

A deep learning based method to automatically identify [miniature EPSPs events](https://en.wikipedia.org/wiki/Excitatory_postsynaptic_potential#Miniature_EPSPs_and_quantal_analysis) from electrophysiology data.

The input of the system is an electrophysiology experiment data and the output is the timestamp and amplitude of the miniature EPSPs events (aka mini peaks). As shown in the graph below it is not trivial to design a handcrafted algorithm to identify the mini peaks because the shape of the signal is the defining discriminator (the amplitude of the signal alone is not sufficient).

![Alt text](media/peaks_detection_graph.png)

The main goal of this toy project is for me to apply what I learnt in the diverse deep learing courses on a real world problem from scratch.

# Installation

## Pytorch

Install pytorch following [the official instructions](https://pytorch.org/get-started/locally/).


## Other dependencies

From the root folder of this repositiory:

```
pip3 install .
```

# Method

TODO explain and pinpoint relevant code:
- input/output data for training
- 1D CNN choice and archi
- overlapping windows
- improve with transformer

# Results

recall, precision, accuracy achieved
