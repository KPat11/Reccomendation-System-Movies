
# 🎬 Movie Recommendation System

This repository demonstrates how to build multiple types of movie recommendation systems using Python and Jupyter Notebooks. It includes implementations of:

- **Content-Based Filtering** using movie descriptions and cosine similarity
- **Popularity-Based Filtering** leveraging IMDb-style vote counts and vote averages
- **Collaborative Filtering** using matrix factorization (SVD) from the `scikit-surprise` library

Each notebook walks through a reproducible, educational pipeline suitable for anyone interested in understanding and implementing recommender systems using real-world movie data.

---

## 📁 Project Structure

- `Content-Based_Filtering.ipynb`  
  Recommends movies based on the textual similarity of their descriptions using TF-IDF and cosine similarity.

- `Content-Based_Filtering_Viewable.py`
  Converted above notebook to a py script to show actual dataset used and code conducted-- too big of a file to display in Github viewer

- `Popularity-Based-Filtering.ipynb`  
  Ranks and recommends movies based on weighted vote average scores (inspired by IMDb’s ranking formula).

- `Collaborative Based Filtering.ipynb`  
  Implements Singular Value Decomposition (SVD) using `scikit-surprise` to predict user ratings for unseen movies.

- `requirements.txt`  
  Contains all the required libraries, including `scikit-surprise`, which requires special installation steps (see below).

---

## 📌 Installation & Setup

This project uses Python 3.11 and is best run in a Conda environment due to the dependencies required by `scikit-surprise`.

### ✅ Step-by-Step Setup

1. **Install Conda** if you haven’t already:  
   [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. **Create a new environment:**
   ```bash
   conda create -n movie-recsys python=3.11
   conda activate movie-recsys
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install `scikit-surprise` manually via Conda (recommended):**
   ```bash
   conda install -c conda-forge scikit-surprise
   ```

   If using Jupyter Notebooks, also install the IPython kernel for this environment:
   ```bash
   python -m ipykernel install --user --name movie-recsys --display-name "Python (movie-recsys)"
   ```

---

## 🧠 Techniques Used

### 📌 Content-Based Filtering
- **TF-IDF Vectorization** on movie overviews/descriptions
- **Cosine Similarity** matrix computation
- Recommends similar movies to a target movie based on description proximity

### 📌 Popularity-Based Filtering
- Based on a **weighted average** formula:
  \[
  \text{Score} = \left( \frac{v}{v + m} \cdot R \right) + \left( \frac{m}{v + m} \cdot C \right)
  \]
  where:
  - \( v \) = number of votes
  - \( m \) = vote threshold (90th percentile)
  - \( R \) = average rating
  - \( C \) = mean vote across all movies

### 📌 Collaborative Filtering (SVD)
- Uses the **Surprise** library
- Applies **SVD decomposition** to a user-item interaction matrix
- Predicts ratings for unseen movies for each user
- Evaluates performance using RMSE and cross-validation

---

## 🚀 How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/KPat11/Reccomendation-System-Movies.git
   cd Reccomendation-System-Movies
   ```

2. Follow the [Installation & Setup](#installation--setup) section.

3. Run the notebooks one by one in JupyterLab or VS Code.

---

## 📈 Example Output

Each notebook prints out sample recommendation lists. For collaborative filtering, predicted ratings for each (user, movie) pair are shown.

---

## 🛠️ Requirements

See `requirements.txt`, which includes:
```
scikit-learn
pandas
numpy
matplotlib
Cython
scikit-surprise
```

> ⚠️ `scikit-surprise` requires C++ compilers or installation via `conda-forge`.

---

## 🧩 Future Improvements

- Add hybrid models combining content and collaborative signals
- Integrate implicit feedback (e.g., watch time, clicks)
- Deploy as a web app using Streamlit or FastAPI

---

## 🧑‍💻 Author

**Ken Patel**  
For questions, reach out via [GitHub](https://github.com/KPat11)
