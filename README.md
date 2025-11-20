#  Movie Recommendation System Using Word2Vec

## Author: Navvya Chaudhary  
## Type: Academic Project  
## Technologies: Python, NLP, Word2Vec, Gensim, Pandas, NumPy

---

## Project Overview  
The **Movie Recommendation System** is a content-based recommendation model built using **Natural Language Processing (NLP)** and **Word2Vec embeddings**.  
The system analyzes movie titles, genres, and plot summaries from the **IMDb Top 1000 dataset** and recommends similar movies based on semantic meaning.  
The recommender runs entirely through a clean, interactive **terminal-based interface**.

---

## Objectives
- To develop a **content-based recommendation system**.  
- To implement **Word2Vec** for converting text into numerical vectors.  
- To provide recommendations based on:
  - Movie Name  
  - Genre  
- To display complete movie details including rating, runtime, certificate, meta score, and overview.

---

## Project Structure

Movie-Recommender-System 

• movie_recommender.py  
• imdb_top_1000.csv  
• Word2Vec_Movie_Model.model  
• README.md  

---

## How the System Works

### **1. Data Loading**  
The IMDb Top 1000 dataset is loaded and cleaned. Columns such as title, genre, overview, runtime, certificate, meta score, and rating are prepared for analysis.

### **2. NLP Preprocessing**  
The text is processed by removing stopwords, converting words to lowercase, and tokenizing sentences into individual words.

### **3. Word2Vec Vectorization**  
A **Word2Vec model trained on the movie dataset** converts movie text into numerical vectors that capture semantic relationships between words.

### **4. Similarity Calculation**  
Cosine similarity is used to compare the vector of one movie with all others to find the closest matches.

### **5. Recommendation Output**  
The system can produce:
- Top 5 similar movies  
- Top 5 movies of a selected genre  
- Detailed information of any movie  

---

## Features  
- Clean and simple **terminal interface**  
- Movie search with detailed metadata  
- Recommendations based on movie content  
- Genre-based movie selection  
- NLP + Word2Vec for intelligent similarity matching  
- Works completely offline  

---

## Dataset Information  
The project uses the **IMDb Top 1000 Movies Dataset**, containing fields such as:  
- Title  
- Year  
- Genre  
- Overview  
- Certificate  
- Runtime  
- Meta Score  
- IMDb Rating  

The dataset is manually cleaned and saved as `imdb_top_1000.csv`.

---

## Installation & Setup

### **1. Install Python 3.11 or above**  
Download from: https://www.python.org/downloads/

### **2. Install required libraries**  
Run in your terminal:


pip3 install gensim numpy pandas

### **3. Add project files**
Place these files in the same directory:
- movie_recommender.py
- imdb_top_1000.csv
- Word2Vec_Movie_Model.model

---

## Usage
To start the Movie Recommender System, run:

bash
python3 movie_recommender.py

Follow the menu options to:
1. Find a movie with full details
2. Get recommendations based on a movie name
3. Get recommendations based on a genre
4. Exit the system

---

## Limitations
- Recommendations depend heavily on the text quality of movie descriptions.
- Dataset contains only 1000 movies, which limits coverage.
- Terminal-based interface only (no GUI).
- Word2Vec embeddings are trained on a small dataset, so semantic representations are limited.
- No collaborative filtering (user-based recommendations are not included).

---

## Future Enhancements
- Build a full **web application** using Streamlit or Flask.
- Train Word2Vec on a larger, richer movie dataset for better accuracy.
- Add advanced NLP models (BERT, Sentence Transformers).
- Add collaborative filtering and hybrid recommendation techniques.
- Implement filters (year range, certificate, runtime, meta score, user preferences).
- Deploy the system online using **Render**, **HuggingFace Spaces**, or **GitHub Pages + API backend**.
- Include movie posters, trailers, and visual UI improvements.

---

## License
This project is created for academic and educational purposes.  
You are free to use or modify it with proper credit to the author.

---

## Acknowledgements
- IMDb dataset contributors  
- Gensim library developers  
- Python open-source community  
- Faculty members for guidance and support  
