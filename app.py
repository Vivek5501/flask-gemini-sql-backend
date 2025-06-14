from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import mysql.connector
import google.generativeai as genai
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.api_core.exceptions import ResourceExhausted

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Database credentials from .env
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

# Sentence embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Few-shot examples
few_shots = [
    {'Question': "What is total revenue if all t-shirts of size M are sold at full price without discounts?",
     'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE size='M';"},
    {'Question': "How many total t-shirts are left in stock?",
     'SQLQuery': "SELECT SUM(stock_quantity) FROM t_shirts;"},
    {'Question': "How many entries of records are present?",
     'SQLQuery': "SELECT COUNT(*) FROM t_shirts;"},
    {'Question': "Tell me all the Nike t-shirts?",
     'SQLQuery': "SELECT * FROM t_shirts WHERE brand='Nike';"},
    {'Question': "What is the average price of all t-shirts?",
     'SQLQuery': "SELECT AVG(price) FROM t_shirts;"},
    {'Question': "Show me the t-shirts available in color red.",
     'SQLQuery': "SELECT * FROM t_shirts WHERE color='Red';"},
    {'Question': "How many t-shirts do we have left for Nike in XS size and white color?",
     'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'"},
    {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
     'SQLQuery': "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'"},
    {'Question': "Show me tshirt ids and their stock quantities.",
     'SQLQuery': "SELECT tshirt_id, stock_quantity FROM t_shirts;"},
    {'Question': "What is the highest discount percentage given?",
     'SQLQuery': "SELECT MAX(pct_discount) FROM discounts;"},
    {'Question': "List all t-shirt IDs with a discount greater than 30 percent.",
     'SQLQuery': "SELECT t_shirt_id FROM discounts WHERE pct_discount > 30;"},
    {'Question': "What is the average discount percentage?",
     'SQLQuery': "SELECT AVG(pct_discount) FROM discounts;"},
    {'Question': "Get all records from the discount table.",
     'SQLQuery': "SELECT * FROM discounts;"}
]

questions = [fs['Question'] for fs in few_shots]
question_embeddings = embedding_model.encode(questions).astype('float32')
index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)

# Prompt for Gemini
prompt = ["""
You are an expert in converting English questions to SQL queries!
The SQL database has two tables: 
1. `t_shirts` with columns - brand, color, size, price, stock_quantity
2. `discounts` with columns - discount_id, t_shirt_id, pct_discount

Please provide only SQL queries without any extra explanation.
Here are some examples:

- What is total revenue if all t-shirts of size M are sold at full price?  
  SQL: SELECT SUM(price * stock_quantity) FROM t_shirts WHERE size='M';

- How many total t-shirts are left in stock?  
  SQL: SELECT SUM(stock_quantity) FROM t_shirts;

- Show me tshirt ids and their stock quantities.  
  SQL: SELECT tshirt_id, stock_quantity FROM t_shirts;

- List all t-shirt IDs with a discount greater than 30 percent.  
  SQL: SELECT t_shirt_id FROM discounts WHERE pct_discount > 30;
          
Please provide only SQL queries without any additional formatting or words.
"""]

def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    try:
        response = model.generate_content([prompt[0], question])
        sql_query = response.text.strip().replace("`", "").replace("’", "'").replace("‘", "'")
        return sql_query
    except ResourceExhausted as e:
        return {"error": "Gemini API quota exceeded. Please try again later."}
    except Exception as e:
        return {"error": "Unexpected error while generating SQL query."}

def read_sql_query(sql):
    conn = None
    try:
        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cur = conn.cursor(dictionary=True)
        cur.execute(sql)
        rows = cur.fetchall()
        return rows
    except mysql.connector.Error as e:
        return {'error': str(e)}
    finally:
        if conn and conn.is_connected():
            conn.close()

# Flask app
app = Flask(__name__)
CORS(app)

@app.route('/get_sql_query', methods=['POST'])
def get_sql_query():
    data = request.json
    question_input = data.get('question')

    if not question_input:
        return jsonify({'error': 'No question provided'}), 400

    user_embedding = embedding_model.encode([question_input]).astype('float32')
    _, indices = index.search(user_embedding, k=1)
    closest_question = questions[indices[0][0]]

    sql_query = get_gemini_response(question_input, prompt)

    if isinstance(sql_query, dict) and 'error' in sql_query:
        return jsonify(sql_query), 429

    sql_result = read_sql_query(sql_query)
    if 'error' in sql_result:
        return jsonify(sql_result), 500

    return jsonify({'sql_query': sql_query, 'sql_result': sql_result}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
