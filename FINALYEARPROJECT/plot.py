import matplotlib.pyplot as plt

# Data from the table
epochs = list(range(1, 21))
training_accuracy = [0.9154, 0.9544, 0.9669, 0.9702, 0.9731, 0.9754, 0.9770, 0.9783, 0.9795, 0.9808, 
                     0.9816, 0.9823, 0.9830, 0.9838, 0.9843, 0.9849, 0.9854, 0.9859, 0.9863, 0.9867]
validation_accuracy = [0.9223, 0.9601, 0.9569, 0.9628, 0.9641, 0.9657, 0.9665, 0.9673, 0.9682, 0.9691, 
                       0.9695, 0.9698, 0.9702, 0.9705, 0.9708, 0.9710, 0.9712, 0.9713, 0.9714, 0.9715]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.grid(True)
plt.legend()
plt.xticks(epochs)
plt.ylim(-0.01, 0.02)
plt.show()