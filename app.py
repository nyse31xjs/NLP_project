import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import os

import utils  

@st.cache_data
def load_data():
    save_path = "data/Nodes/tweet_vectors.pkl"
    with open(save_path, "rb") as f:
        tweet_vector = pickle.load(f)
    id_event_week = pd.read_csv("data/Nodes/id_event_week.csv", sep=',', encoding="utf-8", engine="python")
    toe_week = pd.read_csv("data/Nodes/toe_week.csv", sep=',', encoding="utf-8", engine="python")
    word_week = (
        pd.read_csv("data/Nodes/word_week.csv", sep=',', encoding="utf-8", engine="python")
        .assign(week_start=lambda x: pd.to_datetime(x.week_start))
    )
    df_user = pd.read_csv("data/Nodes/user_st.csv", sep=',', encoding="utf-8", engine="python")
    return id_event_week, toe_week, word_week, tweet_vector, df_user

id_event_week, toe_week, word_week, tweet_vector, df_user = load_data()

@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

st.sidebar.title("Pages")
pages = ["Welcome Page", "Timeline Event", "Users Statistics", "Topic Modeling", "Information Retrieval"]
page = st.sidebar.radio("Go to page...", pages)

st.sidebar.markdown("---")  
st.sidebar.write("Authors: Alexandre Lhuisset, Lucas Miedzyrzecki, Hugo Rameil")
st.sidebar.write("Date: March 2025")

#################### PAGE 0 #####################

if page == pages[0]:
    st.title("Critical Event Tweet Analysis")
    st.write("""
             🌍 Crisis Event Tweet Analysis Dashboard
            Welcome to the Crisis Event Tweet Analysis app! 
            🚀 This interactive dashboard helps analyze social media activity during crisis events such as floods, fires, earthquakes, and other disasters.""")

    st.write("📊 What You Can Do Here...")   
    st.write("📅 Analyze Event Timelines – Understand how discussions evolve over time.")
    st.write("👥 Explore User Statistics – Learn more about the users behind the tweets.")
    st.write("🗂️ Discover Key Topics – Identify important themes using topic modeling.")
    st.write(" 📡 Information Retrieval – Find the most relevant tweets using intelligent filtering and similarity-based ranking.")
        
    st.image("pictures_app/picture.jpg", width=600)
    st.write("Authors: Alexandre Lhuisset, Lucas Miedzyrzecki, Hugo Rameil")
    st.write("Date: March 2025")
    
#################### PAGE 1 #####################

    
if page == pages[1]:
    st.title("Timeline Event")
    
    st.title("Events")

    selected_event = st.selectbox("Choose an event", list(id_event_week.id_event.unique()))
        
    if st.button("Validate event"):
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=id_event_week.query(f'id_event == "{selected_event}"').week_start, y=id_event_week.query(f'id_event == "{selected_event}"').tweet_number, mode='lines+markers', name='lines+markers'))
        
        fig.update_layout(
            title=f"Number of tweets per week for {selected_event}",
            xaxis_title="Week",
            yaxis_title="Number of tweets",
            width=800,
            height=500
        )
        
        st.plotly_chart(fig)
       
    st.title("Type of Events")
 
    selected_toe = st.selectbox("Choose a type of events", list(toe_week.eventType.unique()))
    
    if st.button("Validate ToE"):
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=toe_week.query(f'eventType == "{selected_toe}"').week_start, y=toe_week.query(f'eventType == "{selected_toe}"').tweet_number, mode='lines+markers', name='lines+markers'))
        
        fig.update_layout(
            title=f"Number of tweets per week for {selected_toe}",
            xaxis_title="Week",
            yaxis_title="Number of tweets",
            width=800,
            height=500
        )
        
        st.plotly_chart(fig)
        
    
    st.title("Word trend")

    word = st.text_input("Enter a word", value="colorado")

    if st.button("Validate word"):
        trend = utils.get_word_trend(word_week, word)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend['week_start'], 
            y=trend['contains_word'], 
            mode='lines+markers', 
            name='Occurrences'
        ))

        fig.update_layout(
            title=f"Number of tweets including '{word}' per week",
            xaxis_title="Week",
            yaxis_title="Number of tweets",
            width=800,
            height=500
        )

        st.plotly_chart(fig)
    
#################### PAGE 2 #####################

if page == pages[2]:
    st.title("Users Statistics")

    # tweets_count
    fig_tweets_count = px.histogram(df_user, x='tweets_count', nbins=100, title="Distribution of Tweets Count per User (after removing values above 80th percentile)")

    fig_tweets_count.update_layout(
        xaxis_title='Tweets Count (per user)', 
        yaxis_title='Frequency',
        xaxis=dict(range=[0, 8])  
    )

    st.plotly_chart(fig_tweets_count)
    
    # followers
    percentile_80 = np.percentile(df_user['followers_count'], 80)
    df_user_filtered = df_user[df_user['followers_count'] <= percentile_80]

    fig_followers_count = px.histogram(df_user_filtered, x='followers_count', nbins=100, title="Distribution of Followers Count (after removing values above 80th percentile)")

    fig_followers_count.update_layout(
        xaxis_title='Followers Count', 
        yaxis_title='Frequency',
        xaxis=dict(range=[0, percentile_80])  
    )

    st.plotly_chart(fig_followers_count)

    # isVerified 
    fig_is_verified = px.bar(df_user, x='isVerified', title="Distribution of Verified Accounts")
    fig_is_verified.update_layout(xaxis_title='Is Verified', yaxis_title='Count')
    st.plotly_chart(fig_is_verified)
    
#################### PAGE 3 #####################
    
if page == pages[3]:
    st.title("Topic Modeling")
    st.title("Tweet Topic Modeling with LDA")
    
    num_topics = st.slider("Select number of topics", 2, 10, 3)
    
    if st.button("Identify topics"):
        with st.status("Processing LDA model... Please wait!", expanded=True) as status:

            vectorizer = CountVectorizer(stop_words='english', max_features=5000)
            X = vectorizer.fit_transform(word_week['cleaned_text'])

            lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda_model.fit(X)
            
            words = vectorizer.get_feature_names_out()
            topics = utils.display_topics(lda_model, words)
            
            status.update(label="Processing complete 🚀", state="complete")
            
        st.subheader("Identified Topics:")
        
        for topic, words in topics.items():
            st.write(f"**{topic}:** {words}")


#################### PAGE 4 #####################
    
if page == pages[4]:
    st.title("Information Retrieval")
    
    query = st.text_input("Enter your query")
    
    if st.button("Get top 5 relevant tweets"):
            
        st.write("Top 5 relevant tweets")
        st.dataframe(
                    utils.IR(query, 5)
                )
        