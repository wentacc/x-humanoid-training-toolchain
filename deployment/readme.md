# ROS2 Deployment Guide

A VLA (Vision-Language-Action) deployment example for BrainCo dexterous hand integration with ROS2 system. 

## Steps

### 1. Model Setup
Edit the model path in `ros2_deployment.py`:
```python
model_path = "PATH_TO_YOUR_MODEL"
```

### 2. Start ROS2 Nodes
Launch the required ROS2 nodes for hardware communication.

### 3. Run Policy Inference
Execute the deployment script:
```bash
python ros2_deployment.py
```

## Files Description
- `ros2_deployment.py`: Main deployment script with PolicyAgentNode.
- `action_policy.py`: Policy inference wrapper class.