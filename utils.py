import pandas as pd
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def get_word_trend(df, word, freq='W'):
    df['contains_word'] = df['cleaned_text'].str.contains(word, case=False, na=False)
    trend = df.groupby(pd.Grouper(key='week_start', freq=freq))['contains_word'].sum().reset_index()
    return trend

def display_topics(model, feature_names, num_words=5):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-num_words:]]
        topics[f"Topic {topic_idx + 1}"] = ', '.join(top_words)
    return topics

def preprocess_text(text):
    doc = nlp(text)
    preprocessed_text = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(preprocessed_text)
        
def get_doc_vector(text):
    doc = nlp(text)
    return doc.vector

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)

def get_top_k_tweets(df, query_vector, k: int = 5):
    df['cosine_similarity'] = df['vector'].apply(lambda x: cosine_similarity(query_vector, x))
    df = df.sort_values(by='cosine_similarity', ascending=False)
    top_k_tweets = df.filter(['id', 'text', 'cosine_similarity']).head(k)
    
    return top_k_tweets

def IR(query,k):
    cleaned_query = preprocess_text(query)
    vector_query = get_doc_vector(cleaned_query)
    top_k_tweets = get_top_k_tweets(df_tweet_vector, vector_query, k)
    
    return top_k_tweets