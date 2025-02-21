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

def softmax(x):
	"""Softmax activation for sequence generation"""
	e_x = np.exp(x - np.max(x)) # Prevent overflow

	return e_x / e_x.sum(axis=-1, keepdims=True)


# ---------------------------
# Simple Neural Network Class
# ---------------------------
class NeuralNetwork:
	def __init__(self, hidden_size, vocab_size, learning_rate=0.01):
		np.random.seed(42)  # For reproducibility
		
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate
		self.vocab_size = vocab_size # Output size now matches vocab size
		
		# Weights (random initialization)
		self.weights_input_hidden = None # Placeholder
		self.weights_hidden_output = np.random.randn(hidden_size, vocab_size) * 0.1
		
		# Biases
		self.bias_hidden = None # Will be initialized dynamically
		self.bias_output = np.zeros((1, vocab_size))

	def initialize_weights(self, input_size):
		"""Initialize weights dynamically based on input size"""
		self.weights_input_hidden = np.random.randn(input_size, self.hidden_size) * 0.1
		self.bias_hidden = np.zeros((1, self.hidden_size))

	def forward(self, X, max_length=20):
		""" Forward pass with sequence generation """
		generated_sequence = []
		
		# Dynamically initialize weights if first time or input size changes
		input_size = X.shape[1]
		if self.weights_input_hidden is None or self.weights_input_hidden.shape[0] != input_size:
			self.initialize_weights(input_size)
		
		current_input = X
		self.hidden_output = None  # Store for backpropagation
		self.final_output = None   # Store for backpropagation

		for _ in range(max_length):
			# Input -> Hidden layer
			hidden_input = np.dot(current_input, self.weights_input_hidden) + self.bias_hidden
			hidden_output = np.tanh(hidden_input)

			# Hidden -> Output layer
			final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
			final_output = softmax(final_input) # P robabilities over vocabulary

			# Select next token (greedy search, can be replaced with sampling)
			next_token = np.argmax(final_output)

			# Stop if "End of Sequence" token is generated
			if next_token == self.vocab_size - 1:
				break

			generated_sequence.append(next_token)

			# Shift input and add new token
			next_input_vector = np.roll(current_input, -1, axis=1) # Shfit left by 1
			next_input_vector[0, -1] = next_token # Normalize input
			current_input = next_input_vector

		return generated_sequence

	def backward(self, X, y):
		""" Backpropagation for sequence-sequence learning """
		# Error (Output Layer)
		output_error = self.final_output - y
		output_delta = output_error  # Softmax with cross-entropy

		# Error (Hidden Layer)
		hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
		hidden_delta = hidden_error * (1 - self.hidden_output ** 2) # Derivative of Tanh

		# Update weights and biases
		self.weights_hidden_output -= np.dot(self.hidden_output.T, output_delta) * self.learning_rate
		self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

		self.weights_input_hidden -= np.dot(X.T, hidden_delta) * self.learning_rate
		self.bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

	def train(self, X, y, epochs=5000, batch_size=64):
		print(f"Training Neural Network")
		
		# Initialize weights for the first time
		input_size = X.shape[1]
		self.initialize_weights(input_size)
		
		""" Mini-batch training for stability """
		for epoch in range(epochs):
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
			
			avg_loss = total_loss / num_batches
			if epoch % 100 == 0:
				print(f"Epoch {epoch}, Loss: {avg_loss:.5f}")