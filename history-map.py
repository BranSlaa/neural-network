from dotenv import load_dotenv
import os
import numpy as np
import wikipediaapi
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import random
import itertools
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from the .env file
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Pinecone
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index name and host
index_name = "history-map-index"
index_host = os.getenv("PINECONE_HOST")  # Ensure this is set in your .env file

# Check if the index exists
existing_indexes = [index['name'] for index in pinecone_client.list_indexes()]
print(existing_indexes)
if index_name not in existing_indexes:
	# Create the index if it doesn't exist
	pinecone_client.create_index(
		name=index_name,
		dimension=5,  # Adjust dimension to match your sequence length
		metric="cosine",
		spec=ServerlessSpec(
			cloud="aws",
			region="us-east-1"
		)
	)
	print(f"Created index '{index_name}'.")
else:
	print(f"Index '{index_name}' already exists.")

# Connect to the index using the host
index = pinecone_client.Index(host=index_host)

# Insert vectors into the index with batching
def chunks(iterable, batch_size=200):
	"""A helper function to break an iterable into chunks of size batch_size."""
	it = iter(iterable)
	chunk = tuple(itertools.islice(it, batch_size))
	while chunk:
		yield chunk
		chunk = tuple(itertools.islice(it, batch_size))

vector_dimemsion = 5
vector_count_maximum = 1000

data_generator = map(lambda i: (f'id-{i}', [random.random() for _ in range(vector_dimemsion)]), range(vector_count_maximum))

# Upsert data into Pinecone
for ids_vectors_chunk in chunks(data_generator, batch_size=200):
	index.upsert(vectors=ids_vectors_chunk)

# Example query
query_vector = [random.random() for _ in range(vector_dimemsion)]
response = index.query(
	namespace="ns1",
	vector=query_vector,
	top_k=2,
	include_values=True,
	include_metadata=True
)

print("Query response:", response)

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
	print(f"Fetching Wikipedia text for {title}")
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
	print(f"Cleaning text")
	text = re.sub(r'\[[0-9]*\]', '', text)  # Remove citations
	text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
	text = text.lower()  # Convert to lowercase
	tokens = word_tokenize(text)  # Tokenize text
	tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic characters
	tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
	print(f"Cleaning complete")
	return tokens

# ---------------------------
# Convert Text to Numeric Data
# ---------------------------
def build_vocab(text_tokens):
	print(f"Building vocabulary from {len(text_tokens)} tokens")
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

	def train(self, X, y, epochs=5000, batch_size=64):
		print(f"Training Neural Network")
		best_loss = float('inf')
		paitence = 500 # stop training if loss doesn't improve for 500 epochs
		counter = 0
		
		""" Mini-batch training for stability """
		for epoch in range(epochs):
			print(f"Epoch {epoch} of {epochs}")
			total_loss = 0
			num_batches = len(X) # batch_size

			for i in range(0, len(X), batch_size):
				X_batch = X[i:i+batch_size]
				y_batch = y[i:i+batch_size]

				batch_predictions = self.forward(X_batch)
				self.backward(X_batch, y_batch)

				# Calculate batch loss and accumulate it
				batch_loss = np.mean((y_batch - batch_predictions) ** 2)
				total_loss += batch_loss

			# Comput average loss over all batches
			avg_loss = total_loss / num_batches

			# Check for early stoppage
			if batch_loss < best_loss:
				best_loss = batch_loss
				counter = 0
			else:
				counter += 1

			if counter > paitence:
				print(f"Early stopping at epoch {epoch}")
				break
			
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
	# "Cold_War",
	# "Africa",
	# "Asia",
	# "Europe",
	# "North_America",
	# "South_America",
	# "Australia",
	# "Antarctica",
	# "Continent",
	# "Country",
	# "City",
	# "State",
	# "Province",
	# "County",
	# "Town",
	# "River",
	# "Mountain",
	# "Lake",
	# "Sea",
	# "Ocean",
	# "Desert",
	# "Forest",
	# "Mesopotamia",
	# "Egypt",
	# "Rome",
	# "Greece",
	# "China",
	# "India",
	# "Japan",
	# "Ancient_Egypt",
	# "Ancient_Rome",
	# "Ancient_Greece",
	# "Ancient_China",
	# "Ancient_India",
	# "Ancient_Japan",
	# "Indus_River",
	# "Ganges_River",
	# "Yangtze_River",
	# "Yellow_River",
	# "Nile_River",
	# "Amazon_River",
	# "Mississippi_River",
	# "Missouri_River",
	# "Colorado_River",
	# "Rio_Grande",
	# "Gulf_of_Mexico",
	# "Gulf_of_St._Lawrence",
	# "History_of_China",
	# "History_of_India",
	# "History_of_Japan",
	# "History_of_Rome",
	# "History_of_Greece",
	# "History_of_Egypt",
	# "Chalcolithic",
	# "Bronze_Age",
	# "Iron_Age",
	# "Axial_Age",
	# "Middle_Ages",
	# "Renaissance",
	# "Enlightenment",
	# "Space_Race",
	# "Internet",
	# "Artificial_Intelligence",
	# "Classical_antiquity",
	# "Late_antiquity",
	# "World_War_I",
	# "French_Revolution",
	# "American_Revolution",
	# "Russian_Revolution",
]

wikipedia_text = " ".join([get_wikipedia_text(page, USER_AGENT) for page in pages])

# Clean and tokenize the text
tokens = clean_text(wikipedia_text)

# ---------------------------
# Balance the Dataset by Removing Overrepresented Words
# ---------------------------
word_freq = Counter(tokens)

common_words = set([word for word, count in word_freq.most_common(1000)])
balanced_tokens = [word for word in tokens if word in common_words]

# Build vocabulary and convert text to numeric format
word_to_index, index_to_word = build_vocab(balanced_tokens)
vectorized_text = text_to_vector(balanced_tokens, word_to_index)

# ---------------------------
# Prepare Training Data
# ---------------------------
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
# Upsert Data into Pinecone
# ---------------------------
vector_ids = [f"vec_{i}" for i in range(len(X_train))]
vectors_to_upsert = [{"id": id_, "values": vector.tolist()} for id_, vector in zip(vector_ids, X_train)]

# Upsert vectors into the index
index.upsert(vectors=vectors_to_upsert[:vector_count_maximum], namespace="ns1")

# Example query
query_vector = X_train[0].tolist()  # Use the first training vector as an example
response = index.query(
	namespace="ns1",
	vector=query_vector,
	top_k=2,
	include_values=True,
	include_metadata=False
)

print("Query response:", response)

# ---------------------------
# Evaluate Accuracy
# ---------------------------
def accuracy(predictions, labels):
	predictions = (predictions > 0.5).astype(int)
	return np.mean(predictions == labels) * 100

# ---------------------------
# Train the Neural Network
# ---------------------------
nn = SimpleNeuralNetwork(input_size=sequence_length, hidden_size=30, output_size=1)

print("Training Neural Network on Wikipedia Data...")
nn.train(X_train, y_train, epochs=50, batch_size=256)

preds = nn.forward(X_train)
print(f"Final Training Accuracy: {accuracy(preds, y_train):.2f}%")

# ---------------------------
# Test Prediction
# ---------------------------
def predict_next_word(nn, input_sequence, word_to_index, index_to_word, sequence_length):
	# Ensure input sequence has enough words
	input_sequence = input_sequence[-sequence_length:]

	# Convert words to indices, handling missing words
	input_vector = [word_to_index[word] if word in word_to_index else 0 for word in input_sequence]

	# Pad with zeros if sequence is too short
	while len(input_vector) < sequence_length:
		input_vector.insert(0, 0) # Pretend Padding

	# Convert to numpy array and reshape
	input_vector = np.array(input_vector).reshape(1, sequence_length) / len(word_to_index)
	
	# Get prediction
	prediction = nn.forward(input_vector)

	# Scale prediction back to word index range
	predicted_index = int(prediction[0][0] * len(word_to_index))

	# Get predicted word, handle unknown words
	predicted_word = index_to_word.get(predicted_index, "UNKNOWN")
	
	return predicted_word

# Example prediction
sample_input = tokens[:sequence_length]  # First 5 words from Wikipedia text
print(f"Input: {' '.join(sample_input)}")
predicted_word = predict_next_word(nn, sample_input, word_to_index, index_to_word, sequence_length)
print(f"Predicted next word: {predicted_word}")