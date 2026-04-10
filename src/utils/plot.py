from __future__ import annotations

import matplotlib.pyplot as plt


def plot_training_curves(epoch_losses, epoch_times):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
    plt.title("Biểu đồ độ hội tụ Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss trung bình")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(epoch_times) + 1), epoch_times, marker="o", color="green")
    plt.title("Biểu đồ thời gian huấn luyện")
    plt.xlabel("Epoch")
    plt.ylabel("Thời gian (giây)")
    plt.grid(True)
    plt.show()

    plt.tight_layout()
    plt.show()
