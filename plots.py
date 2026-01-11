import matplotlib.pyplot as plt
import numpy as np

# Daten (Brain entfernt)
categories = ["Chest", "Bone", "Animal", "Baggage"]
cxr = [0.9840, 0.9936, 0.9744, 0.4444]
dino = [0.9782, 0.9987, 0.9870, 0.7765]

x = np.arange(len(categories))
width = 0.35


# Plot
plt.figure(figsize=(6, 4))
plt.bar(x - width/2, cxr, width, label="CXR", color="#00519E")
plt.bar(x + width/2, dino, width, label="DINOv3", color="#7DC6EA")

plt.xticks(x, categories)
plt.ylabel("F1-Score")
plt.xlabel("Datasets")
plt.ylim(0, 1.0)  # optional, aber meist sinnvoll
plt.legend()
plt.ylim(0, 1.05)
#plt.figure(figsize=(8, 5))
plt.tight_layout()
plt.savefig("/home/lisa/Documents/Uni/Module/FM/barchart.png", dpi=300, format="png", bbox_inches="tight")
plt.show()