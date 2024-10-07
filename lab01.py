# Dinh Hoang Viet Phuong - 301123263


# import all necessary libraries
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import numpy as np


# 1. Retrieve and load the mnist_784 dataset of 70,000 instances
# Fetch the mnist_784 dataset
mnist = fetch_openml('mnist_784', version=1)

# Data and target values
X, y = mnist["data"], mnist["target"]


# 2. Display each digit.
# A dictionary to keep track of the first occurrence of each digit
displayed_digits = {}

# Iterate over the dataset to find the first occurrence of each digit
for index, label in enumerate(y):
    if label not in displayed_digits:
        displayed_digits[label] = X.iloc[index].values.reshape(28, 28)
    
    # Break the loop if we've found all digits (0-9)
    if len(displayed_digits) == 10:
        break

# Display each digit
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for ax, (digit, image) in zip(axes, displayed_digits.items()):
    ax.imshow(image, cmap="binary")
    ax.axis("off")
    ax.set_title(digit)

plt.show()


# 3. Use PCA to retrieve the 1st and 2nd principal component and output their explained variance ratio
# Apply PCA and get the first two principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Output the explained variance ratio for the first two components
explained_variance_ratio = pca.explained_variance_ratio_

print(f"Explained variance ratio of 1st principal component: {explained_variance_ratio[0]:.4f}")
print(f"Explained variance ratio of 2nd principal component: {explained_variance_ratio[1]:.4f}")


# 4. Plot the projections of the 1st and 2nd principal component onto a 1D hyperplane
# Plotting the projections onto the 1st principal component
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], [0] * len(X_pca), alpha=0.5, s=1)
plt.title('Projection onto 1st Principal Component')
plt.xlabel('1st Principal Component')
plt.yticks([])

# Plotting the projections onto the 2nd principal component
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 1], [0] * len(X_pca), alpha=0.5, s=1, c='red')
plt.title('Projection onto 2nd Principal Component')
plt.xlabel('2nd Principal Component')
plt.yticks([])

plt.tight_layout()
plt.show()


# 5. Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions
# Apply Incremental PCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

# Split the data into batches and fit the IncrementalPCA instance
for X_batch in np.array_split(X, n_batches):
    inc_pca.partial_fit(X_batch)

# Transform the dataset to 154 dimensions
X_reduced = inc_pca.transform(X)

print(f"Reduced dataset shape: {X_reduced.shape}")


# 6. Display the original and compressed digits from step 5
X = np.array(mnist["data"])

# Use Incremental PCA's inverse_transform to get the compressed (reconstructed) digits
X_reconstructed = inc_pca.inverse_transform(X_reduced)

# Choose a random index for demonstration
index = 999

# Original image
original_image = X[index].reshape(28, 28)

# Reconstructed (compressed) image
reconstructed_image = X_reconstructed[index].reshape(28, 28)

# Display the original and reconstructed images side by side
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Original Image
axes[0].imshow(original_image, cmap="binary")
axes[0].axis("off")
axes[0].set_title("Original Image")

# Reconstructed Image
axes[1].imshow(reconstructed_image, cmap="binary")
axes[1].axis("off")
axes[1].set_title("Reconstructed Image")

plt.tight_layout()
plt.show()






