# CIFAR-10 Image Classification – Two Approaches

This repo demonstrates **two different ways** to train a model for CIFAR-10 image classification using PyTorch:

---

## Files
- **`using-simple-CNN.py`** → Build and train your own CNN from scratch  
- **`using-pretrained-resnet.py`** → Use a pretrained ResNet model and fine-tune it for CIFAR-10

---

## 1. `using-simple-CNN.py` – Build Your Own CNN

### What this does:
- Implements a **custom CNN** with basic layers:
- Trains it **from scratch** on CIFAR-10
- Helps understand the fundamentals of CNNs

### Good for:
- Learning how CNN layers work
- Building intuition about filters, pooling, etc.
- Customizing your own architecture

---

## 2. `using-pretrained-resnet.py` – Fine-tune Pretrained ResNet

### What this does:
- Loads a **ResNet18** model pretrained on **ImageNet**
- Replaces the final classification layer (1000 classes → 10 for CIFAR-10)
- Optionally **freezes earlier layers** to save training time
- Fine-tunes the model on CIFAR-10

### Why use pretrained?
- ResNet has already learned powerful features (edges, textures, shapes) from millions of images
- So, you don’t need to train everything from scratch
- You get **faster training** and often **higher accuracy**

### Good for:
- Leveraging deep architectures without needing lots of data
- Faster and more efficient training
- Real-world deployment scenarios

---

## Summary of Differences

| Feature                     | Simple CNN               | Pretrained ResNet         |
|----------------------------|--------------------------|---------------------------|
| Architecture               | Manually defined CNN     | Deep pretrained ResNet-18 |
| Trained from scratch?      | Yes                      | No (only last layers tuned) |
| Training time              | Slower                   | Faster                    |
| Accuracy                   | Moderate (~70–80%)       | High (~85–90%+)           |
| Customization              | Easy to tweak architecture | Limited to final layers   |
| Good for learning?         | Yes                      | Yes (for transfer learning) |
| Good for real-world use?   | ⚠Needs tuning            | Yes                      |

---