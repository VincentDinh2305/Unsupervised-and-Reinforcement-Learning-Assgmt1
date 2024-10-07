# Dinh Hoang Viet Phuong - 301123263


# import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# 1. Generate Swiss roll dataset
n_samples = 1500
noise = 0.05  # add some noise for variability
X, color = make_swiss_roll(n_samples, noise=noise)


# 2. Plot the resulting generated Swiss roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Swiss Roll Dataset")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()


# 3,4. Use Kernel PCA (kPCA) with linear kernel, a RBF kernel, and a sigmoid kernel
# 3a. Apply kPCA with a linear kernel
kpca_linear = KernelPCA(n_components=2, kernel="linear")
X_kpca_linear = kpca_linear.fit_transform(X)

# Plot the transformation
plt.figure(figsize=(8, 6))
plt.scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Swiss Roll Dataset After kPCA with Linear Kernel")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()

# 3b. Apply kPCA with an RBF kernel
kpca_rbf = KernelPCA(n_components=2, kernel="rbf", gamma=10)
X_kpca_rbf = kpca_rbf.fit_transform(X)

# Plot the transformation
plt.figure(figsize=(8, 6))
plt.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Swiss Roll Dataset After kPCA with RBF Kernel")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()

# 3c. Apply kPCA with a sigmoid kernel
kpca_sigmoid = KernelPCA(n_components=2, kernel="sigmoid", gamma=10, coef0=1)
X_kpca_sigmoid = kpca_sigmoid.fit_transform(X)

# Plot the transformation
plt.figure(figsize=(8, 6))
plt.scatter(X_kpca_sigmoid[:, 0], X_kpca_sigmoid[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Swiss Roll Dataset After kPCA with Sigmoid Kernel")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()


# 5. Using kPCA and a kernel of your choice, apply Logistic Regression for classification. 
# se GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification 
# accuracy at the end of the pipeline. Print out the best parameters found by GridSearchCV.
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, color, test_size=0.2, random_state=42)

# Set up the pipeline
pipeline = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])

# Use GridSearchCV
param_grid = {
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["linear", "rbf", "sigmoid"]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_search.fit(X_train, y_train > np.median(y_train))  # We're treating the classification as a binary problem (above or below median)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Test the model on the test set and print the accuracy
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test > np.median(y_test))
print("Test set accuracy with best parameters: {:.2f}".format(test_score))


# 6. Plot the results from using GridSearchCV in step 5
# Create a grid in the original space for plotting the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50),
                         np.linspace(z_min, z_max, 50))

# Transform the grid using the best kPCA and predict using logistic regression
Z = grid_search.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# Since the result is in the original 3D space, let's pick one of the dimensions (e.g., z-dimension) to show the decision boundary
plt.figure(figsize=(10, 7))
plt.contourf(xx[:, :, 0], yy[:, :, 0], Z[:, :, 0], alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train > np.median(y_train), marker='o', edgecolors='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test > np.median(y_test), marker='s', s=60, edgecolors='k')
plt.title("Decision Boundary with Best kPCA Parameters")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()










