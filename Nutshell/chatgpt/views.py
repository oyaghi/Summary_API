# myapp/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import random
import re
from sentence_transformers import SentenceTransformer, util
from transformers import BartTokenizer, BartForConditionalGeneration

# Load models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = 'facebook/bart-base'
bart_tokenizer = BartTokenizer.from_pretrained(model_name)
bart_model = BartForConditionalGeneration.from_pretrained(model_name)

# Load data
df = pd.read_excel('Dataset_v2.xlsx')
df.columns = ['Question', 'Answer']
questions = df['Question'].tolist()
question_embeddings = sentence_model.encode(questions)

# Helper functions
def contains_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return bool(url_pattern.search(text))

def get_response(input_text, num_return_sequences=5):
    inputs = bart_tokenizer.encode(input_text, return_tensors='pt', truncation=True)
    outputs = bart_model.generate(inputs, max_length=100, num_beams=5, num_return_sequences=num_return_sequences, early_stopping=True)
    decoded_outputs = [bart_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return decoded_outputs

def get_similarity(user_query):
    query_embedding = sentence_model.encode(user_query)
    similarities = util.cos_sim(query_embedding, question_embeddings)
    if similarities.shape[1] == 0:
        return None, None
    most_similar_index = similarities.argmax().item()
    similarity_score = similarities[0][most_similar_index].item()
    return similarity_score, most_similar_index

def get_answer(most_similar_index):
    if 0 <= most_similar_index < len(df):
        answer = df['Answer'].iloc[most_similar_index]
        if contains_url(answer):
            return answer
        else:
            paraphrased_answer = get_response(answer)
            return random.choice(paraphrased_answer)
    else:
        return "No valid answer found."

# API View
class QuestionAnswerView(APIView):
    def post(self, request):
        user_input = request.data.get('question')
        if not user_input:
            return Response({"error": "No question provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        similarity_score, most_similar_index = get_similarity(user_input)
        if most_similar_index is None:
            return Response({"error": "No valid similarity found"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        if similarity_score >= 0.5:
            answer = get_answer(most_similar_index)
            return Response({"similarity_score": similarity_score, "answer": answer}, status=status.HTTP_200_OK)
        else:
            return Response({"message": "I don't have an answer for the question you have provided. Maybe my friend ChatGPT could help you."}, status=status.HTTP_200_OK)
