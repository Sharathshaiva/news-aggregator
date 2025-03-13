# News Aggregator Bot

This project extracts the latest news articles, summarizes them, and provides a Q&A interface using Llama3.

## Features
✅ Extracts the latest news articles from NDTV.
✅ Summarizes the content in a meaningful story format.
✅ Enables Q&A from the summarized news content.
✅ User-friendly interface using Streamlit.

## Installation
### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/news-aggregator.git
```

### Step 2: Navigate to the Project Directory
```bash
cd news-aggregator
```

### Step 3: Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate       # For Windows
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Add Environment Variables
Create a `.env` file in the project folder and add the following:
```
HF_TOKEN=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage
### Step 1: Run the Streamlit Application
```bash
streamlit run app.py
```

### Step 2: Using the Interface
- Click **Start** to extract NDTV's latest news articles.
- Click **Summarize the content** to generate meaningful summaries.
- Use the **Q&A FROM the above NEWS** button to enable querying from the extracted news.

## Project Structure
```
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
```

## Requirements
- Python 3.9 or higher
- Required libraries are listed in `requirements.txt`
