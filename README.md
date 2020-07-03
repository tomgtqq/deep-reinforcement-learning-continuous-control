[//]: # (Image References)
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# deep-reinforcement-learning-continuous-control
Distributed Training robotic-arm can move to target locations using actor-critic DDPG.
### Introduction

For this project, Training a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

#### Real World
Having multiple copies of the same agent sharing experience can accelerate learning.
<div align="center">
<img src="assets/robotic_arms.gif" height="200" width="400">
</div>

#### Simulation Environment
Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables simulations to serve as environments for training intelligent agents.
For this project, work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.
<div align="center">
<img src="assets/result.gif" height="200" width="400">
</div>

### Getting Started
The Project is for Udacity Deep Reinforcement learning nd. 

### Download the Unity Environment
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

### Dependencies
1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```

2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

### Running

```
conda activate drlnd

jupyter notebook 
```
Then select Continuous_Control.ipynb and running 

### Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

### Built With

* [Udacity](https://github.com/udacity/deep-reinforcement-learning) - Udacity Deep Reinforcement learning nd
* [Unity](https://github.com/Unity-Technologies/ml-agents/tree/master/docs) - Unity ML-Agents Toolkit Documentation
### Resources
* [Continuous Control With Deep Reinforcement Learning(DDPG)](https://arxiv.org/pdf/1707.06347.pdf)
* [Proximal Policy Optimization Algorithms(PPO)](https://arxiv.org/pdf/1707.06347.pdf)
* [Asynchronous Methods for Deep Reinforcement Learning(A3C)](https://arxiv.org/pdf/1602.01783.pdf)
* [Distributed Distributional Deterministic Policy Gradients(D4PG)](https://openreview.net/pdf?id=SyZipzbCb)
* [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/pdf/1604.06778.pdf)
* [Openai Baselines](https://openai.com/blog/openai-baselines-ppo/)
### Authors

* **Tom Ge** - *Fullstack egineer* - [github profile](https://github.com/tomgtqq)

### License

This project is licensed under the MIT License
