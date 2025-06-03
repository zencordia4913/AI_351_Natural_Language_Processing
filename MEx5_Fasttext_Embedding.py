
"""
Machine Exercise # 5
By: Jeryl Salas

We are tasked to create fasttext embedding model using the 1000 documents we've been using since before. 
We are also asked to use a character length for the words to be 3 characters long (n=3). 
We will then compate with the fasttext embedding model in the library of gensim
"""


"""---------------------------------1. LOADING IMPORTS---------------------------------------------"""

import os # Used for opening and writing text documents
import json # Used for loading JSON formatted documents
import pandas as pd # Used for structuring dataframes
import random # Select files at random
from nltk.tokenize import RegexpTokenizer, sent_tokenize # Used for tokenization of sentences
from typing import Iterator # For faster generation of tokens
from collections import defaultdict # For faster creation and updating of dictionaries
from tqdm import tqdm # For displaying of progress bar
import time # For checking the duration of the training model
from gensim.models import FastText # Used as a comparison for our model. Its a requirement in our MEx
import numpy as np # For matrix operations
import pycuda.driver as cuda # Used for loading our CUDA driver. I had to use a GPU device for this MEx. Training with CPU's are so slow
import pycuda.autoinit # Helps setup the CUDA context
from pycuda.compiler import SourceModule # Used for defining our sourceModule. It's the part of the code that we want to parallelize

"""---------------------------------2. LOADING DATASET---------------------------------------------"""

def load_dataset(folder_path, n):
    """
    Load JSON data files into pandas DataFrame for training and testing.
    """
    files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
    selected_files = random.sample(files, n)
    selected_files_test = selected_files[:2]
    selected_files_train = selected_files[2:]

    def load_files(file_list):
        data = []
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data.extend(json.load(file))  
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading {file_name}: {e}")
        return pd.DataFrame(data)

    train_df = load_files(selected_files_train)
    test_df = load_files(selected_files_test)

    return train_df['text'], test_df['text']


def replace_characters(text: str) -> str:
    """
    Replace specific special characters in the input text.
    """
    replacement_rules = {'<': '', '>': '', '/': "", '--': ''}
    for symbol, replacement in replacement_rules.items():
        text = text.replace(symbol, replacement)
    return text


def generate_tokens(paragraph: str) -> Iterator[str]:
    """
    Tokenize sentences and append '[END]' token at the end of each sentence.
    """
    word_tokenizer = RegexpTokenizer(r'[-\'\w]+')
    for sentence in sent_tokenize(paragraph):
        tokenized_sentence = word_tokenizer.tokenize(sentence)
        if tokenized_sentence:
            tokenized_sentence.append('[END]')
            yield tokenized_sentence


def preprocess(folder_path):
    """
    Main preprocessing function to load and process text data.
    """
    train_set, test_set = load_dataset(folder_path, 750)
    train_tokenized_sentences = []
    test_tokenized_sentences = []

    for text in train_set:
        cleaned_text = replace_characters(text.lower())
        for tokenized_sentence in generate_tokens(cleaned_text):
            train_tokenized_sentences.append(tokenized_sentence)

    for text in test_set:
        cleaned_text = replace_characters(text.lower())
        for tokenized_sentence in generate_tokens(cleaned_text):
            test_tokenized_sentences.append(tokenized_sentence)

    return train_tokenized_sentences, test_tokenized_sentences


"""---------------------------------3. PREPROCESS DATA---------------------------------------------"""
print(f"Loading dataset...")
print(f"Preprocessing data...")
train_tokenized_sentences, test_tokenized_sentences = preprocess(r'C:\Users\Jeryl Salas\Documents\AI 351\MEx 2 Tokenizer\coleridgeinitiative-show-us-the-data\train')
print(f"Tokenized {len(train_tokenized_sentences)} sentences")



"""---------------------------------4. NGRAM GENERATION---------------------------------------------"""
def get_ngrams(word, n=3):
    """
    Returns n-grams for the given word. Depending on what n value was set
    """
    word = f"<{word}>"  
    return [word[i:i+n] for i in range(len(word)-n+1)]

# Just as an example
word = "apple"
print(f"Testing ngram generation...")
print(f"At n = 3, apple: {get_ngrams(word)}")  # ['<ap', 'app', 'ppl', 'ple', 'le>']



"""---------------------------------5. SOURCE MODULE---------------------------------------------"""
"""
The loss, sigmoid, and gradient calculations are performed in parallel across multiple threads. Each thread 
processes a specific dimension of the word vectors and context vectors. The operation is designed for Stochastic 
Gradient Descent (SGD) optimization. The function updates the word vector and context vector based on the gradient, 
taking into account whether the current pair is a positive or negative sample, as indicated by the boolean flag `is_positive`. 
The n-gram vectors are integrated into the updates, influencing the word vector during training.
"""




mod = SourceModule("""
__global__ void parallel_dot_and_update_with_negatives(
    float *word_vectors, float *context_vectors, float *neg_context_vectors,
    float *ngram_vectors, float *gradients, int vector_size, float learning_rate, int is_positive) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vector_size) {
        // Calculating the dot product with n-gram vectors. This is the computation for the score
        float dot_product = word_vectors[idx] * context_vectors[idx];
        float ngram_dot_product = word_vectors[idx] * ngram_vectors[idx]; // Integrate n-gram influence

        // Computation for the sigmoid           
        float sigmoid_val = 1.0 / (1.0 + expf(-dot_product));
        float gradient;

        // If we're using positive samples           
        if (is_positive) {
            gradient = sigmoid_val - 1.0;  // Positive pair gradient (push dot product towards 1)
        } else {
            gradient = sigmoid_val;  // Negative pair gradient (push dot product towards 0)
        }

        // Update word vectors based from gradients
        word_vectors[idx] -= learning_rate * gradient * (context_vectors[idx] + ngram_dot_product);
        context_vectors[idx] -= learning_rate * gradient * word_vectors[idx];

        // If we're using negative samples                 
        if (!is_positive) {
            // Updating the negative context vectors
            neg_context_vectors[idx] -= learning_rate * gradient * word_vectors[idx];
        }
    }
}
""")                   

print(f"Loading CUDA driver...")
parallel_dot_and_update = mod.get_function("parallel_dot_and_update_with_negatives")



"""---------------------------------6. TRAINING MODEL---------------------------------------------"""
"""
We then create a function that performs parallelized training of word vectors using the Skip-gram model with negative sampling, 
with our GPU device. We first collect positive word-context pairs from the sentence within a specified 
window size, then randomly generates negative samples. The word, context, and n-gram vectors are allocated and 
transferred to the GPU, where the parallel computation of dot products, sigmoid, and gradient updates are executed 
using CUDA kernels. Positive and negative samples are processed separately. Once the GPU updates are complete, the updated 
vectors are transferred back to the CPU and used to update the word and context vectors in the vocabulary. The allocated 
GPU memory is then freed.
"""



def train_skipgram_gpu_parallel_fasttext(sentence, word_vectors, context_vectors, ngram_vectors, window_size=5, learning_rate=0.01, num_negative_samples=5):
    batch_positive_samples = []
    batch_context_samples = []
    batch_negative_samples = []
    
    vocabulary = list(word_vectors.keys())
    
    # Collecting positive samples
    for i, target_word in enumerate(sentence):
        start = max(0, i - window_size)
        end = min(len(sentence), i + window_size + 1)
        context_words = sentence[start:i] + sentence[i+1:end]

        for context_word in context_words:
            batch_positive_samples.append(target_word) # Adding positive word-context pairs
            batch_context_samples.append(context_word)

    # Initialize random batch of negative samples
    batch_negative_samples = [random.choice(vocabulary) for _ in range(num_negative_samples * len(batch_positive_samples))]

    # Compute grid sizes for CUDA processing
    vector_size = len(word_vectors[batch_positive_samples[0]])
    block_size = 128
    grid_size = (vector_size + block_size - 1) // block_size

    # Allocating GPU memory for word, context, ngram vectors, and negative samples
    word_vecs_gpu = cuda.mem_alloc(len(batch_positive_samples) * vector_size * 4)
    context_vecs_gpu = cuda.mem_alloc(len(batch_context_samples) * vector_size * 4)
    neg_context_vecs_gpu = cuda.mem_alloc(len(batch_negative_samples) * vector_size * 4)
    ngram_vecs_gpu = cuda.mem_alloc(len(batch_positive_samples) * vector_size * 4)
    gradients_gpu = cuda.mem_alloc(len(batch_positive_samples) * vector_size * 4)
    
    # Copying word, context, and negative vectors to GPU. Making sure that they're using float32
    # CPU --> GPU
    selected_word_vectors = np.array([word_vectors[word] for word in batch_positive_samples], dtype=np.float32)
    d_word_vecs_gpu = cuda.mem_alloc(selected_word_vectors.nbytes)
    cuda.memcpy_htod(d_word_vecs_gpu, selected_word_vectors)

    selected_context_vectors = np.array([word_vectors[word] for word in batch_context_samples], dtype=np.float32)
    d_context_vecs_gpu = cuda.mem_alloc(selected_context_vectors.nbytes)
    cuda.memcpy_htod(d_context_vecs_gpu, selected_context_vectors)
    
    selected_negative_vectors = np.array([word_vectors[word] for word in batch_negative_samples], dtype=np.float32)
    d_neg_context_vecs_gpu = cuda.mem_alloc(selected_negative_vectors.nbytes)
    cuda.memcpy_htod(d_neg_context_vecs_gpu, selected_negative_vectors)

    # Lookup of vectors from our ngram vectors vocabulary so we don't have generate Ngrams each iteration which makes training slower.
    batch_ngram_vectors = []
    for word in batch_positive_samples:
        if word in ngram_vectors:
            batch_ngram_vectors.append(ngram_vectors[word])
        else:
            batch_ngram_vectors.append(np.zeros(vector_size, dtype=np.float32)) # If in case the ngram doesn't exist which is near impossible.

    batch_ngram_vectors = np.array(batch_ngram_vectors, dtype=np.float32)

    ngram_vecs_gpu = cuda.mem_alloc(batch_ngram_vectors.nbytes) # Allocate ngram vectors into GPU
    cuda.memcpy_htod(ngram_vecs_gpu, batch_ngram_vectors) # Copying ngram vecs into GPU.

    # Launching kernel for positive samples 
    parallel_dot_and_update(
        d_word_vecs_gpu, d_context_vecs_gpu, d_neg_context_vecs_gpu, ngram_vecs_gpu, gradients_gpu, 
        np.int32(vector_size), np.float32(learning_rate), np.int32(1), 
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )

    # Launching kernel for negative samples
    parallel_dot_and_update(
        d_word_vecs_gpu, d_context_vecs_gpu, d_neg_context_vecs_gpu, ngram_vecs_gpu, gradients_gpu, 
        np.int32(vector_size), np.float32(learning_rate), np.int32(0), 
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )

    # Initialize first an empty Numpy array to accomodate the float32 byte transfer from GPU --> CPU
    updated_word_vectors = np.empty_like(selected_word_vectors)
    updated_context_vectors = np.empty_like(selected_context_vectors)

    # Copy vectors GPU --> CPU
    cuda.memcpy_dtoh(updated_word_vectors, d_word_vecs_gpu)
    cuda.memcpy_dtoh(updated_context_vectors, d_context_vecs_gpu)

    # Updating the original word_vectors and context_vectors via LookUp on their respective vocabularies
    for i, word in enumerate(batch_positive_samples):
        word_vectors[word] = updated_word_vectors[i]

    for i, word in enumerate(batch_context_samples):
        context_vectors[word] = updated_context_vectors[i]

    # Free GPU memory
    word_vecs_gpu.free()
    context_vecs_gpu.free()
    neg_context_vecs_gpu.free()
    ngram_vecs_gpu.free()
    gradients_gpu.free()
    

 
"""---------------------------------7. CACHING RESULTS---------------------------------------------"""
"""
We write the results of word, context, and ngram vectors on .txt files for checking
"""


def cache_vectors(word_vectors, context_vectors, ngram_vectors, file_location_1, file_location_2, file_location_3): 
    """
    Used for caching vectors on a .txt file for analysis. Since we're not using .ipnyb file. Can't get CUDA to work on a jupyter notebook.
    """
    
    with open(file_location_1, 'w', encoding='utf-8') as f:
        f.write("---------Word Vectors----------\n")
        for word, vector in word_vectors.items():
            f.write(f"{word}\t{','.join(map(str, vector))}\n")
            
    with open(file_location_2, 'w', encoding='utf-8') as f:
        f.write("\n-------Context Vectors-------\n")
        for context_word, context_vector in context_vectors.items():
            f.write(f"{context_word}\t{','.join(map(str, context_vector))}\n")
            
    with open(file_location_3, 'w', encoding='utf-8') as f:
        f.write("\n-------N-gram Vectors--------\n")
        for ngram, ngram_vector in ngram_vectors.items():
            f.write(f"{ngram}\t{','.join(map(str, ngram_vector))}\n")



# Initialize vocabulary for word, context, ngram embedding
vocabulary = defaultdict(int)

# Define epochs and embedding dim
epochs = 5
embedding_dim = 100

print(f"Building vocabulary...")
for sentence in train_tokenized_sentences:
    for word in sentence:
        ngrams = get_ngrams(word)  
        for ngram in ngrams:
            vocabulary[ngram] += 1  
        vocabulary[word] += 1  

# Initialize random vectors for word, context, and n-gram embeddings
print(f"Initializing vectors...")        
word_vectors = {word: np.random.uniform(-0.1, 0.1, embedding_dim).astype(np.float32) for word in vocabulary}
context_vectors = {word: np.random.uniform(-0.1, 0.1, embedding_dim).astype(np.float32) for word in vocabulary}
ngram_vectors = {}
for word in vocabulary:
    for ngram in get_ngrams(word):
        if ngram not in ngram_vectors:  
            ngram_vectors[ngram] = np.random.uniform(-0.1, 0.1, embedding_dim).astype(np.float32)


# Training loop
print(f"\n________________________TRAINING_______________________________\n")
print(f"Will perform training on {epochs} epochs and {len(train_tokenized_sentences)} tokenized sentences")

pbar = tqdm(total=epochs*len(train_tokenized_sentences)) # For display of progress bar
start_time = time.time()
total_sentences = len(train_tokenized_sentences)

for epoch in range(epochs):
    #print(f"Epoch {epoch + 1}/{epochs}")
    for i, sentence in enumerate(train_tokenized_sentences):
        #print(f"Processing sentence {i+1}/{len(train_tokenized_sentences)}")
        train_skipgram_gpu_parallel_fasttext(sentence, word_vectors, context_vectors, ngram_vectors)
        pbar.update(1)

        # Calculating remaining time for display
        elapsed_time = time.time() - start_time
        total_processed = (epoch * total_sentences) + (i + 1)
        total_estimated_time = (elapsed_time / total_processed) * (epochs * total_sentences)
        remaining_time = total_estimated_time - elapsed_time
        minutes, seconds = divmod(remaining_time, 60)
        pbar.set_description(f"Epoch: {epoch + 1}/{epochs}, Sentence: {i + 1}/{total_sentences}, Remaining: {int(minutes)} mins {int(seconds)} secs")

# Calculate duration of the training 
end_time = time.time()
time_taken = end_time - start_time
print(f"\n______________________DONE TRAINING_______________________________\n")

minutes, seconds = divmod(time_taken, 60)
print(f"Training for {epochs*len(train_tokenized_sentences)} sentences completed in {int(minutes)} minutes and {int(seconds)} seconds.")

# Defining file locations for caching results
file_1 = "C:/Users/Jeryl Salas/Documents/AI 351/MEx 5 Fasttext Embedding/word_vector_cache.txt"
file_2 = "C:/Users/Jeryl Salas/Documents/AI 351/MEx 5 Fasttext Embedding/context_vector_cache.txt"
file_3 = "C:/Users/Jeryl Salas/Documents/AI 351/MEx 5 Fasttext Embedding/ngram_vector_cache.txt"
file_4 = "C:/Users/Jeryl Salas/Documents/AI 351/MEx 5 Fasttext Embedding/comparison_vector_cache.txt"

print(f"Caching vectors...")
cache_vectors(word_vectors, context_vectors, ngram_vectors, file_1, file_2, file_3)




"""---------------------------------8. TESTING BY COMPARISON WITH GENSIM IMPLEMENTATION---------------------------------------------"""
"""
We now compare embeddings of our model with gensim.fasttext. Both models used the same training set. And will now test their embeddings on the testing set
"""



def embed_sentences(sentences, custom_word_vectors, gensim_model=None, file_location="embedding_comparison.txt", embedding_dim=100):
    
    mean_differences = []
    with open(file_location, 'w', encoding='utf-8') as f:
        
        f.write("___________COMPARISON FOR CUSTOM AND GENSIM IMPLEMENTATION___________________\n\n")
 
        for sentence in sentences:
            tokens = sentence
            custom_vector = np.mean([custom_word_vectors.get(token, np.zeros(embedding_dim)) for token in tokens], axis=0)

            if gensim_model:
                gensim_vector = gensim_model.wv[sentence] 

            # Writing the results on file
            f.write(f"Sentence: {sentence}\n")
            f.write(f"Custom: {','.join(map(str, custom_vector))}\n")
            
            if gensim_vector is not None:
                f.write(f"Gensim: {','.join(map(str, gensim_vector))}\n")
                
                # Computing mean absolute difference between two vectors
                mean_difference = np.mean(np.abs(custom_vector - gensim_vector))
                mean_differences.append(mean_difference)
                f.write(f"Mean Difference: {mean_difference}\n")
            else:
                f.write("Gensim: No embedding found\n")
                f.write("Mean Difference: N/A\n")
            
            f.write("\n")  
     
    return mean_differences
            

print(f"Comparing current model to gensim fasttext...")
start_time = time.time()
model = FastText(sentences=train_tokenized_sentences, vector_size=100, window=5, min_count=1, sg=1, epochs=5)
end_time = time.time()
time_taken = end_time - start_time
model.save("fasttext_model.bin")
gensim_model = FastText.load("fasttext_model.bin")
minutes, seconds = divmod(time_taken, 60)
print(f"Training using gensim model completed in {int(minutes)} minutes and {int(seconds)} seconds.")

# Using the test tokenized sentences. We now perform the comparison
mean_diffs = embed_sentences(test_tokenized_sentences, word_vectors, gensim_model, file_4)
print(f"\n\nPrinted results in the comparison_vector_cache.txt...")
#print(f"Mean differences between the model and gensim model: \n")
#print(mean_diffs) 
