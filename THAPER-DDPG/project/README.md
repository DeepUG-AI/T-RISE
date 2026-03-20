# Tunnel-Oriented Multi-UAV Obstacle Avoidance in AirSim

This repository contains the simulation code for multi-UAV obstacle avoidance in tunnel environments using a TensorFlow 1.x based DDPG reinforcement learning framework on AirSim.

## 1. Environment

Tested environment:

- Python 3.7
- Windows 10/11
- CUDA 10.0
- AirSim 1.8.1
- TensorFlow-GPU 1.15.0
- Unreal Engine with AirSim plugin

## 2. Main Dependencies

The project was tested with the following main packages:

- airsim==1.8.1
- msgpack==0.6.2
- msgpack-rpc-python==0.4.1
- numpy==1.19.5
- opencv-python==4.5.5.64
- Pillow==8.4.0
- protobuf==3.20.3
- tensorboard==1.15.0
- tensorflow-estimator==1.15.1
- tensorflow-gpu==1.15.0
- tornado==4.5.3
- xlwt==1.3.0

## 3. Installation

Install from the requirements file:

```bash
pip install -r requirements.txt
```

Or install the required packages manually:

```bash
pip install airsim==1.8.1
pip install msgpack==0.6.2
pip install msgpack-rpc-python==0.4.1
pip install numpy==1.19.5
pip install opencv-python==4.5.5.64
pip install Pillow==8.4.0
pip install protobuf==3.20.3
pip install tensorboard==1.15.0
pip install tensorflow-estimator==1.15.1
pip install tensorflow-gpu==1.15.0
pip install tornado==4.5.3
pip install xlwt==1.3.0
```

## 4. Project Structure

```text
project/
├── algorithms/
│   ├── actor.py
│   ├── critic.py
│   ├── tunnel_ddpg.py
│   ├── replay/
│   │   ├── per_memory.py
│   │   └── sum_tree.py
│   └── modules/
│       ├── state_adapter.py
│       ├── reward_shaper.py
│       ├── exploration_scheduler.py
│       ├── delay_compensator.py
│       ├── sensor_noise.py
│       └── human_prior_demonstration.py
├── envs/
│   └── tunnel_drone_env.py
├── configs/
│   └── tunnel_ddpg.yaml
├── train.py
├── test.py
├── README.md
├── LICENSE
└── __init__.py
```

## 5. Training

Before training, make sure:

1. AirSim is installed correctly.
2. The Unreal/AirSim simulation environment is running.
3. The API connection between Python and AirSim is available.

Run training with:

```bash
python train.py
```

## 6. Testing

Before testing, please do the following first:

1. Find the saved best model files from training.
2. Copy these best model files into the `data_coord` folder.
3. Then run the testing script.

Run testing with:

```bash
python test.py
```

## 7. Notes

- Please make sure the AirSim simulator is started before running training or testing.
- The test script reads the required model files from the `data_coord` folder.
- If the model files are not placed in `data_coord`, the test process may fail.
- GPU acceleration in our setup requires CUDA 10.0.

## 8. License

This project is released under the LICENSE included in this repository.
