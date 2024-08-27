# Balancing Energy Efficiency in Sensor networks with Multi-Agent Reinforcement Learning (BEES-MARL)

This code repository accompanies the paper titled "Balancing Energy Preservation and Performance in Energy-harvesting Sensor Networks." The repository provides the implementation of the BEES-MARL algorithm along with the general environment as described in the paper. The proposed BEES-MARL algorithm which optimizes energy efficiency in IoT networks by leveraging correlations between observations from multiple energy-harvesting sensors. The code includes the simulation environment for general settings used to evaluate the approach's effectiveness in reducing redundant transmissions, minimizing latency, and preventing outages while maintaining task performance.

## Usage

In `main.py`, you can select one of the available strategies: `greedy`, `alternating`, `learned (DQN)`, or `BEES-MARL (MARL)`. Within the same file, you can also modify various parameters such as the energy required to transmit a bit, bandwidth, time-step, and more.

## Links
SensorLab: https://sensorlab.ijs.si/

## Licensing
Copyright 2024 Jernej Hribar
The code in this project is licensed under MIT license.

## Acknowledgement
The research leading to these results was supported in part by the JST PRESTO Grant No. JPMJPR1854, JSPS KAKENHI Grant No. JP21H03427, and JSPS International Research Fellow Grant No. PE20723. This work was also funded in part by the European Regional Development Fund through the SFI Research Centres Programme under Grant No. 13/RC/2077\_P2 SFI CONNECT and by the SFI-NSFC Partnership Programme Grant No. 17/NSFC/5224. 
