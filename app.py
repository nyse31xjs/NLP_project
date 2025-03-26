import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import utils  


@st.cache_data
def load_data():
    id_event_week = pd.read_csv("/Users/hugorameil/Desktop/Code/GitHub/NLP_project/data/Nodes/id_event_week.csv", sep=',', encoding="utf-8", engine="python")
    toe_week = pd.read_csv("/Users/hugorameil/Desktop/Code/GitHub/NLP_project/data/Nodes/toe_week.csv", sep=',', encoding="utf-8", engine="python")
    tweet_vector = pd.read_csv("/Users/hugorameil/Desktop/Code/GitHub/NLP_project/data/Nodes/tweet_vector.csv", sep=';', encoding="utf-8", engine="python")
    return id_event_week, toe_week, tweet_vector

id_event_week = load_data()[0]
toe_week = load_data()[1]
tweet_vector = load_data()[2]

nlp = utils.nlp  


st.sidebar.title("Pages")
pages = ["Welcome page", "Timeline Event", "Information Retrieval"]
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

    st.write("📊 What You Can Do Here")   
    st.write("📅 Analyze Event Timelines – Understand how discussions evolve over time.")
    st.write("🗂️ Discover Key Topics – Identify important themes using word clusters and embeddings.")
    st.write(" 📡 Information Retrieval – Find the most relevant tweets using intelligent filtering and similarity-based ranking.")
    
    # insert a picture 
    
    st.image("/Users/hugorameil/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Code/GitHub/NLP_project/pictures_app/picture.jpg", width=600)
    st.write("Authors: Alexandre Lhuisset, Lucas Miedzyrzecki, Hugo Rameil")
    st.write("Date: March 2025")
#################### PAGE 1 #####################

    
if page == pages[1]:
    st.title("Timeline Event")
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
    

    
#################### PAGE 2 #####################

    
if page == pages[2]:
    st.title("Information Retrieval")
    
    query = st.text_input("Enter your query")
    
    if st.button("Get top 5 relevant tweets"):
        
        cleaned_query = utils.preprocess_text(query)
        query_vector = utils.get_doc_vector(cleaned_query)  
            
        st.write("Top 5 relevant tweets")
        st.dataframe(
                    utils.get_top_5_tweets(tweet_vector, query_vector).filter(['id', 'text', 'cosine_similarity'])      
                )

        
        
