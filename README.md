# 🧩 My ComfyUI Custom Node Pack

![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge)
![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-Compatible-green?style=for-the-badge)

---

## 📌 Overview

This repository contains a collection of **custom nodes for ComfyUI**, designed to simplify workflows and add new creative / utility features.  
Each node is plug-and-play and appears directly inside ComfyUI after installation.

---

## 🎯 Features

- Easy plug-and-play installation  
- Organized node categories  
- Clear controllable parameters  
- Designed for efficiency and expanded capability  

---

## 📥 Installation

### **Option 1 — Git Clone (Recommended)**

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/YourUserName/YourNodePack.git
```

### **Option 2 — Manual Install**

1. Click Code → Download ZIP
2. Extract into:
```bash
ComfyUI/custom_nodes/YourNodePack/
```

## ⭐ Custom Nodes Overview
A complete list of all custom nodes included in this pack.  
Use the summary table below to quickly jump to any node’s documentation.

---

## 📚 Summary Table
| Node Name | Short Description | Link |
|---|---|---|
| Dual Ease Cosine Scheduler | One sentence describing this node. | [Jump to Section](#yournodenamehere) |
| YourNextNode | One sentence describing this node. | [Jump to Section](#yournextnode) |
| AnotherNode | One sentence describing this node. | [Jump to Section](#anothernode) |
| ... | ... | ... |

---

## 🧩 Node Documentation

### Dual Ease Cosine Scheduler
*This node is a custom scheduler I made that needs further testing. It's been tested using FID (Fréchet Inception Distance) along ablation to find the rough best default values.*

**Settings**
| Setting Name | Description |
|-------------|-------------|
| steps | controls how many steps will be made |
| sigma_max | controls the maximum sigma (max noise) |
| min_sigma | controls the minimum sigma (min noise) |

**Example Usage**
```text
(Optional) Describe how or when this node is typically used.
