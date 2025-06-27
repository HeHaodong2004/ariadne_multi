# ariadne_multi

This repository contains a **multi-agent exploration** system based on the original [ARiADNE2](https://github.com/marmotlab/ARiADNE2) codebase.

It extends the single-agent framework into a flexible multi-robot setting for collaborative exploration tasks.

---

## 🔧 How to Use

- The number of agents can be controlled by modifying the `N_AGENTS` variable in [`parameter.py`](./parameter.py).

---

## 🧠 Code Structure

- **`agent.py`**  
  All logic related to individual robot behavior is encapsulated here.  
  To modify or customize the actions and policies of a single robot, edit this file.

- **`node_manager.py`**  
  Handles map structure and exploration node management.  
  To change how the map is parsed, updated, or managed, edit this file.

---

## 📁 Other Components

- `driver.py` – Main entry point to launch training 
- `runner.py` – High-level rollout controller  
- `model.py`, `env.py` – Core RL models and simulation environment  
- `utils.py`, `sensor.py`, etc. – Supporting utilities and sensor modeling  

---


## 🚧 Planned Improvements

- **Inpainting module integration**  
  Incorporate a generative map inpainting model to improve exploration under occlusion or uncertainty.

- **Communication module**  
  Add a message-passing system to enable inter-agent communication and coordination under partial observability.
