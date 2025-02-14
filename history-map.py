import numpy as np
import wikipediaapi
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

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
# Wikipedia Data Fetching
# ---------------------------
def get_wikipedia_text(title, user_agent):
	wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language='en')
	page = wiki.page(title)

	if page.exists():
		return page.text
	else:
		print(f"Page '{title}' not found.")
		return ""

# ---------------------------
# Text Preprocessing
# ---------------------------
def clean_text(text):
	text = re.sub(r'\[[0-9]*\]', '', text)  # Remove citations
	text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
	text = text.lower()  # Convert to lowercase
	tokens = word_tokenize(text)  # Tokenize text
	tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic characters
	tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
	return tokens

# ---------------------------
# Convert Text to Numeric Data
# ---------------------------
def build_vocab(text_tokens):
	vocab = list(set(text_tokens))  # Unique words
	word_to_index = {word: i for i, word in enumerate(vocab)}
	index_to_word = {i: word for word, i in word_to_index.items()}
	return word_to_index, index_to_word

def text_to_vector(text_tokens, word_to_index):
	return np.array([word_to_index[word] for word in text_tokens if word in word_to_index])

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

	def train(self, X, y, epochs=5000, batch_size=16):
		""" Mini-batch training for stability """
		for epoch in range(epochs):
			total_loss = 0
			for i in range(0, len(X), batch_size):
				X_batch = X[i:i+batch_size]
				y_batch = y[i:i+batch_size]

				batch_predictions = self.forward(X_batch)
				self.backward(X_batch, y_batch)

				# Calculate batch loss and accumulate it
				batch_loss = np.mean((y_batch - batch_predictions) ** 2)
				total_loss += batch_loss
			
			# Print loss every 500 epochs
			if epoch % 500 == 0:
				avg_loss = total_loss / (len(X) / batch_size) # Normalize loss by batch size
				print(f"Epoch {epoch}, Loss: {avg_loss:.5f}")
			
			print(f"Epoch {epoch} complete")

# ---------------------------
# Load and Preprocess Wikipedia Data
# ---------------------------
USER_AGENT = "History Map (iam@thethird.dev)"

# Fetch multiple related Wikipedia pages
pages = [
	"History_of_the_world",
	"Human",
	"Prehistory",
	"Ancient_civilizations",
	"Industrial_Revolution",
	"World_War_II",
	"Cold_War",
	"Africa",
	"Asia",
	"Europe",
	"North_America",
	"South_America",
	"Australia",
	"Antarctica",
	"Continent",
	"Country",
	"City",
	"State",
	"Province",
	"County",
	"Town",
	"River",
	"Mountain",
	"Lake",
	"Sea",
	"Ocean",
	"Desert",
	"Forest",
	"Mesopotamia",
	"Egypt",
	"Rome",
	"Greece",
	"China",
	"India",
	"Japan",
	"Ancient_Egypt",
	"Ancient_Rome",
	"Ancient_Greece",
	"Ancient_China",
	"Ancient_India",
	"Ancient_Japan",
	"Indus_River",
	"Ganges_River",
	"Yangtze_River",
	"Yellow_River",
	"Nile_River",
	"Amazon_River",
	"Mississippi_River",
	"Missouri_River",
	"Colorado_River",
	"Rio_Grande",
	"Gulf_of_Mexico",
	"Gulf_of_St._Lawrence",
	"History_of_China",
	"History_of_India",
	"History_of_Japan",
	"History_of_Rome",
	"History_of_Greece",
	"History_of_Egypt",
	"Chalcolithic",
	"Bronze_Age",
	"Iron_Age",
	"Axial_Age",
	"Middle_Ages",
	"Renaissance",
	"Enlightenment",
	"Space_Race",
	"Internet",
	"Artificial_Intelligence",
	"Classical_antiquity",
	"Late_antiquity",
	"World_War_I",
	"French_Revolution",
	"American_Revolution",
	"Russian_Revolution",
]

wikipedia_text = " ".join([get_wikipedia_text(page, USER_AGENT) for page in pages])

# Clean and tokenize the text
tokens = clean_text(wikipedia_text)
print(tokens)

# Build vocabulary and convert text to numeric format
word_to_index, index_to_word = build_vocab(tokens)
vectorized_text = text_to_vector(tokens, word_to_index)

# ---------------------------
# Prepare Training Data
# ---------------------------
# We create simple "next word prediction" data:
# Each input is a sequence of `sequence_length` words, and the target is the next word.
sequence_length = 5  # Each input has 5 words

X_train, y_train = [], []
for i in range(len(vectorized_text) - sequence_length):
	X_train.append(vectorized_text[i:i+sequence_length])  # Input words
	y_train.append(vectorized_text[i+sequence_length])    # Next word

X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, 1)

# Normalize input values (for better training)
X_train = X_train / len(word_to_index)
y_train = y_train / len(word_to_index)

# ---------------------------
# Evaluate Accuracy
# ---------------------------
def accuracy(predictions, labels):
	predictions = (predictions > 0.5).astype(int)
	return np.mean(predictions == labels) * 100

# ---------------------------
# Train the Neural Network
# ---------------------------
nn = SimpleNeuralNetwork(input_size=sequence_length, hidden_size=10, output_size=1)

print("Training Neural Network on Wikipedia Data...")
nn.train(X_train, y_train, epochs=5000, batch_size=32)

preds = nn.forward(X_train)
print(f"Final Training Accuracy: {accuracy(preds, y_train):.2f}%")

# ---------------------------
# Test Prediction
# ---------------------------
def predict_next_word(nn, input_sequence, word_to_index, index_to_word):
	input_vector = np.array([word_to_index[word] for word in input_sequence if word in word_to_index])
	input_vector = input_vector.reshape(1, -1) / len(word_to_index)
	
	prediction = nn.forward(input_vector)
	predicted_index = int(prediction[0][0] * len(word_to_index))  # Rescale prediction to vocabulary size
	predicted_word = index_to_word.get(predicted_index, "UNKNOWN")
	
	return predicted_word

# Example prediction
sample_input = tokens[:sequence_length]  # First 5 words from Wikipedia text
print(f"Input: {' '.join(sample_input)}")
predicted_word = predict_next_word(nn, sample_input, word_to_index, index_to_word)
print(f"Predicted next word: {predicted_word}")
