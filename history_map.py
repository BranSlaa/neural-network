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
import concurrent.futures
from neural_network import SimpleNeuralNetwork
import time

# Load environment variables from the .env file
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Initialize Pinecone
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index name and host
index_name = "history-map-index"
index_host = os.getenv("PINECONE_HOST")  # Ensure this is set in your .env file

vector_dimemsion = 5
vector_count_maximum = 1000

# Check if the index exists
existing_indexes = [index['name'] for index in pinecone_client.list_indexes()]
print(existing_indexes)
if index_name not in existing_indexes:
	# Create the index if it doesn't exist
	pinecone_client.create_index(
		name=index_name,
		dimension=vector_dimemsion,  # Adjust dimension to match your sequence length
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
	
def fetch_all_wikipedia_pages(pages, user_agent):
	wikipedia_texts = []
	
	# Use ThreadPoolExecutor to parallelize requests
	with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
		future_to_page = {executor.submit(get_wikipedia_text, page, user_agent): page for page in pages}
		
		for future in concurrent.futures.as_completed(future_to_page):
			wikipedia_texts.append(future.result())

	return " ".join(wikipedia_texts)

# ---------------------------
# Text Preprocessing
# ---------------------------
def clean_text(text):
	print(f"Cleaning text")
	text = re.sub(r'\[[0-9]*\]', '', text)  # Remove citations
	text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
	tokens = word_tokenize(text)  # Tokenize text
	tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove non-alphabetic characters and Stopwords
	return tokens

# ---------------------------
# Convert Text to Numeric Data
# ---------------------------
def build_vocab(text_tokens):
	print(f"Building vocabulary from {len(text_tokens)} tokens")
	start_time = time.time()  # Start timing
	vocab = list(set(text_tokens))  # Unique words
	word_to_index = {word: i for i, word in enumerate(vocab)}
	index_to_word = {i: word for word, i in word_to_index.items()}
	print(f"Vocabulary built in {time.time() - start_time:.2f} seconds")  # End timing
	return word_to_index, index_to_word

def text_to_vector(text_tokens, word_to_index):
	return np.array([word_to_index[word] for word in text_tokens if word in word_to_index])

# ---------------------------
# Load and Preprocess Wikipedia Data
# ---------------------------
USER_AGENT = "History Map (iam@thethird.dev)"

# Fetch multiple related Wikipedia pages
pages = [
	"Human_history",
	"Human",
	"Prehistory",
	"Ancient_civilizations",
	"Industrial_Revolution",
	"World_War_II",
	"Cold_War",
	"Africa",
	"United_States",
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
	"Presidential_system",
	"Federal_government_of_the_United_States",
	"Constitution_of_the_United_States",
	"Liberal_democracy",
	"United_States_Congress",
	"United_States_Senate",
	"United_States_House_of_Representatives",
	"United_States_Supreme_Court",
	"United_States_Constitution",
	"United_States_Bill_of_Rights",
	"President_of_the_United_States",
	"Vice_President_of_the_United_States",
]

wikipedia_text = fetch_all_wikipedia_pages(pages, USER_AGENT)

# Clean and tokenize the text
tokens = clean_text(wikipedia_text)

# ---------------------------
# Balance the Dataset by Removing Overrepresented Words
# ---------------------------
word_freq = Counter(tokens)

common_words = set([word for word, count in word_freq.most_common(1000)])
balanced_tokens = [word for word in tokens if word in common_words]

# Build vocabulary and convert text to numeric format
word_to_index, index_to_word = build_vocab(tokens)
print(f"Vocabulary built.")
vectorized_text = text_to_vector(tokens, word_to_index)
print(f"Vectorized text.")

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

print(f"Training data prepared.")
# Function to check if vectors exist in the index
def check_existing_vectors(index, vector_ids, namespace="ns1", batch_size=100):
	existing_vectors = []
	
	for i in range(0, len(vector_ids), batch_size):
		batch = vector_ids[i:i+batch_size]
		response = index.fetch(ids=batch, namespace=namespace)

		if response and response.vectors:
			for vector_id in batch:
				if vector_id in response.vectors:
					existing_vectors.append(response.vectors[vector_id].values)

	return existing_vectors

# Generate vector IDs
vector_ids = [f"vec_{i}" for i in range(len(X_train))]

# Check for existing vectors
# print(f"Checking {len(vector_ids)} vectors in Pinecone...")
# existing_vectors = check_existing_vectors(index, vector_ids)
# print(f"Existing vectors: {existing_vectors}")
# if existing_vectors:
# 	print("Using existing vectors from Pinecone.")
# 	X_train = np.array(existing_vectors)
# else:
	# print("No existing vectors found. Calculating and upserting new vectors.")
	# Normalize input values (for better training)
X_train = X_train / len(word_to_index)
y_train = y_train / max(y_train)

	# Upsert vectors into the index
	# vectors_to_upsert = [{"id": id_, "values": vector.tolist()} for id_, vector in zip(vector_ids, X_train)]
	# index.upsert(vectors=vectors_to_upsert[:vector_count_maximum], namespace="ns1")

# Example query
# query_vector = X_train[0].tolist()  # Use the first training vector as an example
# response = index.query(
# 	namespace="ns1",
# 	vector=query_vector,
# 	top_k=2,
# 	include_values=True,
# 	include_metadata=False
# )

# print("Query response:", response)

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

num_samples = min(len(X_train), len(y_train))

X_train = X_train[:num_samples]
y_train = y_train[:num_samples]

y_train = y_train * max(word_to_index.values()) # undo normalization

print("Training Neural Network on Wikipedia Data...")
nn.train(X_train, y_train, epochs=500, batch_size=64)

preds = nn.forward(X_train).reshape(-1, 1)
print(f"Final Training Accuracy: {accuracy(preds, y_train):.2f}%")

# ---------------------------
# Test Prediction
# ---------------------------
def predict_next_word(nn, input_sequence, word_to_index, index_to_word, sequence_length):
	# Ensure input sequence has enough words
	input_sequence = input_sequence[-sequence_length:]

	# Convert words to indices, handling missing words
	input_vector = [word_to_index.get(word, 0) for word in input_sequence]

	# Pad with zeros if sequence is too short
	while len(input_vector) < sequence_length:
		input_vector.insert(0, 0) # Pretend Padding

	# Convert to numpy array and reshape
	input_vector = np.array(input_vector).reshape(1, sequence_length) / len(word_to_index)
	
	# Get prediction
	prediction = nn.forward(input_vector)

	# Scale prediction back to word index range
	predicted_index = int(prediction[0][0] * max(word_to_index.values()))
	
	return index_to_word.get(predicted_index, "UNKNOWN")

# Example prediction
sample_input = ["Who", "is", "the", "president", "of", "the", "United", "States"]
print(f"Input: {' '.join(sample_input)}")
predicted_word = predict_next_word(nn, sample_input, word_to_index, index_to_word, sequence_length)
print(f"Predicted next word: {predicted_word}")