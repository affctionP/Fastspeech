import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_excel("chat_v0.0.xlsx")
df=df.dropna()
questions=df['He_trans'].tolist()
answers=df['She_trans'].tolist()
# Initialize and fit the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_answer(user_input):

    # Transform user input into TF-IDF vector
    user_input_vector = vectorizer.transform([user_input])
    # Compute cosine similarity between input and all questions
    similarities = cosine_similarity(user_input_vector, question_vectors)
    closest_index = similarities.argmax()  # Find the most similar question
    return answers[closest_index]  # Return the corresponding answer



def chat_create(user_id,chat_history):
    import os

    file_path = f'users_history/chat_{user_id}.txt'  # Change this to your file's path

    # Check if the file exists
    with open(file_path, "a",encoding='utf-8') as file:
        file.write(chat_history)

