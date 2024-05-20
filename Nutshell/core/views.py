# myapp/views.py
import json
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
from django.views.decorators.csrf import csrf_exempt
from scipy.special import softmax
from scipy.sparse.csgraph import connected_components



# Your helper functions and logging setup
def degree_centrality_scores(similarity_matrix, threshold=None, increase_power=True):
    if not (threshold is None or isinstance(threshold, float) and 0 <= threshold < 1):
        raise ValueError("'threshold' should be a floating-point number from the interval [0, 1) or None")
    
    if threshold is None:
        markov_matrix = create_markov_matrix(similarity_matrix)
    else:
        markov_matrix = create_markov_matrix_discrete(similarity_matrix, threshold)
    
    scores = stationary_distribution(markov_matrix, increase_power=increase_power, normalized=False)
    return scores

def _power_method(transition_matrix, increase_power=True, max_iter=10000):
    eigenvector = np.ones(len(transition_matrix))
    if len(eigenvector) == 1:
        return eigenvector
    
    transition = transition_matrix.transpose()
    for _ in range(max_iter):
        eigenvector_next = np.dot(transition, eigenvector)
        if np.allclose(eigenvector_next, eigenvector):
            return eigenvector_next
        eigenvector = eigenvector_next
        if increase_power:
            transition = np.dot(transition, transition)
    return eigenvector_next

def connected_nodes(matrix):
    _, labels = connected_components(matrix)
    groups = []
    for tag in np.unique(labels):
        group = np.where(labels == tag)[0]
        groups.append(group)
    return groups

def create_markov_matrix(weights_matrix):
    n_1, n_2 = weights_matrix.shape
    if n_1 != n_2:
        raise ValueError("'weights_matrix' should be square")
    row_sum = weights_matrix.sum(axis=1, keepdims=True)
    if np.min(weights_matrix) <= 0:
        return softmax(weights_matrix, axis=1)
    return weights_matrix / row_sum

def create_markov_matrix_discrete(weights_matrix, threshold):
    discrete_weights_matrix = np.zeros(weights_matrix.shape)
    ixs = np.where(weights_matrix >= threshold)
    discrete_weights_matrix[ixs] = 1
    return create_markov_matrix(discrete_weights_matrix)

def stationary_distribution(transition_matrix, increase_power=True, normalized=True):
    n_1, n_2 = transition_matrix.shape
    if n_1 != n_2:
        raise ValueError("'transition_matrix' should be square")
    distribution = np.zeros(n_1)
    grouped_indices = connected_nodes(transition_matrix)
    for group in grouped_indices:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        distribution[group] = eigenvector
    if normalized:
        distribution /= n_1
    return distribution

# Main summarization function
@api_view(["POST"])
@csrf_exempt
def Summarization(request):
    nltk.download('punkt')
    
    data = json.loads(request.body)
    document = data.get('document', '')

        # Check if document is provided
    if not document:
        return Response({"error": "No document provided."}, status=400)

    # Split the document into sentences
    sentences = nltk.sent_tokenize(document)
    print("Num sentences:", len(sentences))

    # Load the sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Parse the input document from the request data
    document = request.data.get('document', '')
    
    if not document:
        return Response({"error": "No document provided"}, status=status.HTTP_400_BAD_REQUEST)

    # Split the document into sentences
    sentences = nltk.sent_tokenize(document)
    
    # Compute the sentence embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute the pair-wise cosine similarities
    cos_scores = util.cos_sim(embeddings, embeddings).numpy()

    # Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

    # Sort the sentences by centrality scores
    most_central_sentence_indices = np.argsort(-centrality_scores)

    # Extract the top 5 most central sentences
    summary_sentences = [sentences[idx] for idx in most_central_sentence_indices[:5]]

    # Join the summary sentences into a single summary string
    summary = ' '.join(summary_sentences)

    return Response({"summary": summary}, status=status.HTTP_200_OK)
