import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_mfcc(audio_file, sr=8000, n_mfcc=13, hop_length=512):
    y, _ = librosa.load(audio_file, sr=sr, mono=True)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

def compute_similarity(feature_matrix1, feature_matrix2):
    # If the two feature matrices have different lengths, truncate the longer one
    min_len = min(feature_matrix1.shape[1], feature_matrix2.shape[1])
    feature_matrix1 = feature_matrix1[:, :min_len]
    feature_matrix2 = feature_matrix2[:, :min_len]
    
    # Compute average cosine similarity for all columns (time frames)
    similarities = [cosine_similarity(feature_matrix1[:, i].reshape(1, -1), 
                                      feature_matrix2[:, i].reshape(1, -1))[0][0] 
                    for i in range(min_len)]
    return np.mean(similarities)

# Extract MFCC features
master_mfcc = extract_mfcc('assets/Pudhu Vellai - Violin 1_1.wav')
live_mfcc = extract_mfcc('assets/Pudhu Vellai - Violin 2_1.wav')

# Compute similarity
similarity_score = compute_similarity(master_mfcc, live_mfcc)
print(f'Similarity Score: {similarity_score}') 
print(f'Similarity Score out of 10 :', format(similarity_score*10,".2f")) 


