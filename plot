import matplotlib.pyplot as plt

# Define data
dev_acc = 0.6217125382262997
test_acc = 0.6124737210932025

# Create bar plot
plt.bar(['DEV', 'TEST'], [dev_acc, test_acc],label = "RoBERTa-large")

# Add labels and title
plt.xlabel('Data')
plt.ylabel('Accuracy')


dev_accuracy=0.6217125382262997
test_accuracy= 0.6377014716187807
plt.bar(['DEV_', 'TEST_'], [dev_accuracy, test_accuracy],label = "RoBERTa-base")
plt.xlabel('Data')
plt.ylabel('Accuracy')

plt.legend()

# Display plot
plt.show()
