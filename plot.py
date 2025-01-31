import matplotlib.pyplot as plt

# Plotting a figure for accuracy in % vs. epoch
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(test_acc_list, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Progress")
plt.legend()
plt.show()
