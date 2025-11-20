import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from gensim.models import Word2Vec
except Exception as e:
    print("Error importing gensim. Install gensim (see README).")
    print(e)
    sys.exit(1)

CSV_NAME = "imdb_top_1000.csv"
W2V_NAME = "Word2Vec_Movie_Model.model"
STOPWORDS = {"the", "a", "an", "and", "of", "in", "on", "for", "to", "with", "is", "are", "as", "by", "at", "from", "this", "that", "it"}

def locate_file(name):
    p = Path(name)
    if p.exists():
        return p
    icloud = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Documents" / name
    if icloud.exists():
        return icloud
    raise FileNotFoundError(f"{name} not found.")

def load_data():
    path = locate_file(CSV_NAME)
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(' ', '_')
    if "Series_Title" in df.columns:
        df = df.rename(columns={
            "Series_Title": "title",
            "Overview": "overview",
            "Genre": "genre",
            "Released_Year": "year",
            "IMDB_Rating": "rating"
        })
    df['title'] = df['title'].astype(str)
    df['overview'] = df.get('overview', "").fillna("")
    df['genre'] = df.get('genre', "").fillna("")
    df['year'] = df.get('year', "")
    df['Certificate'] = df.get('Certificate', "N/A")
    df['Runtime'] = df.get('Runtime', "N/A")
    df['Meta_score'] = df.get('Meta_score', "N/A")
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        df['rating'] = None
    df['combined'] = (df['title'].astype(str) + " " + df['genre'].astype(str) + " " + df['overview'].astype(str)).str.lower()
    return df.reset_index(drop=True)

def tokenize(text):
    words = re.findall(r"\w+", str(text).lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

def get_doc_vector(text, model):
    tokens = tokenize(text)
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=float)
    return np.mean(vecs, axis=0)

def build_vectors(df, w2v_model):
    if w2v_model is None:
        raise RuntimeError("Word2Vec model not loaded")
    vecs = []
    for t in df['combined']:
        vecs.append(get_doc_vector(t, w2v_model))
    return np.vstack(vecs)

def find_title(df, name):
    name_low = name.strip().lower()
    exact = df[df['title'].str.lower() == name_low]
    if not exact.empty:
        return int(exact.index[0]), exact.iloc[0]['title']
    contains = df[df['title'].str.lower().str.contains(name_low)]
    if not contains.empty:
        return int(contains.index[0]), contains.iloc[0]['title']
    words = [w for w in re.findall(r"\w+", name_low) if w not in STOPWORDS]
    for w in words:
        cand = df[df['title'].str.lower().str.contains(re.escape(w))]
        if not cand.empty:
            return int(cand.index[0]), cand.iloc[0]['title']
    return None, None

def cosine_sim(a, M):
    a_norm = np.linalg.norm(a)
    M_norm = np.linalg.norm(M, axis=1)
    if a_norm == 0:
        return np.zeros(M.shape[0])
    sims = M.dot(a) / (M_norm * a_norm + 1e-9)
    return sims

def recommend_by_movie(df, vecs, idx, top_n=5):
    sims = cosine_sim(vecs[idx], vecs)
    order = np.argsort(sims)[::-1]
    recs = []
    for i in order:
        if i == idx:
            continue
        recs.append((int(i), float(sims[i])))
        if len(recs) >= top_n:
            break
    return recs

def recommend_by_genre(df, genre_query, top_n=5):
    q = genre_query.strip().lower()
    subset = df[df['genre'].astype(str).str.lower().str.contains(q)]
    if subset.empty:
        return []
    if subset['rating'].notna().any():
        subset = subset.sort_values(by='rating', ascending=False)
    return [int(i) for i in subset.head(top_n).index]

def print_movie_short(df, idx):
    row = df.iloc[idx]
    print("\nTitle:", row.get('title', 'N/A'))
    print("Year:", row.get('year', 'N/A'))
    print("Genre:", row.get('genre', 'N/A'))
    print("Certificate:", row.get('Certificate', 'N/A'))
    print("Runtime:", row.get('Runtime', 'N/A'))
    print("Meta Score:", row.get('Meta_score', 'N/A'))
    print("IMDB Rating:", row.get('rating', 'N/A'))
    ov = str(row.get('overview', ''))[:400]
    if len(str(row.get('overview', ''))) > 400:
        ov += "..."
    print("Overview:", ov)
    print("-" * 60)

def main():
    print("-----------------------------------")
    print("MOVIE RECOMMENDER SYSTEM")
    print("Owner: Navvya Chaudhary")
    print("-----------------------------------")
    print()
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(e)
        return
    try:
        w2v = Word2Vec.load(str(locate_file(W2V_NAME)))
    except Exception as e:
        print("Failed to load Word2Vec model:", e)
        return
    vecs = build_vectors(df, w2v)
    while True:
        print("\nChoose an option:")
        print("1. Find movie by name (show details)")
        print("2. Recommend based on movie name")
        print("3. Recommend by genre")
        print("4. Quit")
        choice = input("Enter your choice (1-4): ").strip()
        if choice == "1":
            name = input("Enter movie name: ")
            idx, matched = find_title(df, name)
            if idx is None:
                print("Movie not found.")
            else:
                print_movie_short(df, idx)
        elif choice == "2":
            name = input("Enter seed movie name: ")
            idx, matched = find_title(df, name)
            if idx is None:
                print("Movie not found.")
            else:
                recs = recommend_by_movie(df, vecs, idx, top_n=5)
                print("\nRecommendations based on:", matched)
                for rank, (i, score) in enumerate(recs, start=1):
                    print(f"{rank}. {df.iloc[i]['title']}  (score {score:.3f})")
                    print(f"   {df.iloc[i].get('genre','')} | {df.iloc[i].get('year','')} | Rating: {df.iloc[i].get('rating','')}")
                    print(f"   Certificate: {df.iloc[i].get('Certificate','N/A')} | Runtime: {df.iloc[i].get('Runtime','N/A')} | Meta Score: {df.iloc[i].get('Meta_score','N/A')}")
        elif choice == "3":
            g = input("Enter genre (e.g., Drama, Action): ")
            ids = recommend_by_genre(df, g, top_n=5)
            if not ids:
                print("No movies found for that genre.")
            else:
                print(f"\nTop movies in genre '{g}':")
                for i in ids:
                    print("-", df.iloc[i]['title'], "|", df.iloc[i].get('year',''), "| Rating:", df.iloc[i].get('rating',''))
                    print(f"   Certificate: {df.iloc[i].get('Certificate','N/A')} | Runtime: {df.iloc[i].get('Runtime','N/A')} | Meta Score: {df.iloc[i].get('Meta_score','N/A')}")
        elif choice == "4":
            print("Exiting... Thank you for using the Movie Recommender System.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

