import numpy as np

# ---------------------------
# Activation Functions
# ---------------------------
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)  # Derivative of sigmoid for backpropagation

def relu(x):
	return np.maximum(0, x)

def relu_derivative(x):
	return np.where(x > 0, 1, 0)  # Derivative of ReLU

# ---------------------------
# Simple Neural Network Class
# ---------------------------
class SimpleNeuralNetwork:
	def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
		np.random.seed(42)  # For reproducibility
		
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate
		
		# Weights (random initialization)
		self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
		self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
		
		# Biases
		self.bias_hidden = np.zeros((1, hidden_size))
		self.bias_output = np.zeros((1, output_size))

	def forward(self, X):
		""" Forward pass with Leaky ReLU activation function """
		# Input -> Hidden layer
		self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
		self.hidden_output = np.maximum(0.01 * self.hidden_input, self.hidden_input) # Leaky ReLU

		# Hidden -> Output layer
		self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
		self.final_output = sigmoid(self.final_input)

		return self.final_output

	def backward(self, X, y):
		""" Backpropagation algorithm to adjust weights and biases """
		# Error (Output Layer)
		output_error = self.final_output - y
		output_delta = output_error * sigmoid_derivative(self.final_output)

		# Error (Hidden Layer)
		hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
		hidden_delta = hidden_error * np.where(self.hidden_output > 0, 1, 0.01) # Leaky ReLU derivative

		# Update weights and biases
		self.weights_hidden_output -= np.dot(self.hidden_output.T, output_delta) * self.learning_rate
		self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

		self.weights_input_hidden -= np.dot(X.T, hidden_delta) * self.learning_rate
		self.bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

	def train(self, X, y, epochs=5000, batch_size=64):
		print(f"Training Neural Network")
		best_loss = float('inf')
		patience = 500  # stop training if loss doesn't improve for 500 epochs
		counter = 0
		
		""" Mini-batch training for stability """
		for epoch in range(epochs):
			print(f"Epoch {epoch} of {epochs}")
			total_loss = 0
			num_batches = len(X) // batch_size  # Calculate the number of full batches

			for i in range(0, len(X), batch_size):
				X_batch = X[i:i+batch_size]
				y_batch = y[i:i+batch_size]

				# Ensure the batch sizes match
				if X_batch.shape[0] != y_batch.shape[0]:
					continue  # Skip this batch if sizes don't match

				batch_predictions = self.forward(X_batch)
				self.backward(X_batch, y_batch)

				# Calculate batch loss and accumulate it
				batch_loss = np.mean((y_batch - batch_predictions) ** 2)
				total_loss += batch_loss

			# Compute average loss over all batches
			avg_loss = total_loss / num_batches

			# Check for early stoppage
			if batch_loss < best_loss:
				best_loss = batch_loss
				counter = 0
			else:
				counter += 1

			if counter > patience:
				print(f"Early stopping at epoch {epoch}")
				break
			
			# Print loss every 500 epochs
			if epoch % 500 == 0:
				print(f"Epoch {epoch}, Loss: {avg_loss:.5f}")
			
			print(f"Epoch {epoch} complete")