from dotenv import load_dotenv
import os
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from lida import TextGenerationConfig
load_dotenv()

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


# Text generation configuration
TEXT_GEN_CONFIG = TextGenerationConfig(n=1,model="text-bison@001")

# Visualization configuration
VIZ_CONFIG = {
    "n": 1,
    "temperature": 0.2,
    "use_cache": True
}

# SQL examples for few-shot learning
SQL_EXAMPLES = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {"input": "Find all albums for the artist 'AC/DC'.", "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');"},
    {"input": "List all tracks in the 'Rock' genre.", "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');"},
    {"input": "Find the total duration of all tracks.", "query": "SELECT SUM(Milliseconds) FROM Track;"},
    {"input": "List all customers from Canada.", "query": "SELECT * FROM Customer WHERE Country = 'Canada';"},
]

POSTGRESQL_EXAMPLES = [
    {"input": "List all employees.", "query": "SELECT * FROM employees;"},
    {"input": "Find all orders for the customer 'ACME Corp'.", "query": "SELECT * FROM orders WHERE customer_id = (SELECT customer_id FROM customers WHERE company_name = 'ACME Corp');"},
    {"input": "List all products in the 'Electronics' category.", "query": "SELECT * FROM products WHERE category_id = (SELECT category_id FROM categories WHERE category_name = 'Electronics');"},
    {"input": "Calculate the total value of all orders.", "query": "SELECT SUM(order_total) FROM orders;"},
    {"input": "List all suppliers from Germany.", "query": "SELECT * FROM suppliers WHERE country = 'Germany';"},
    {"input": "Find the average price of products in each category.", "query": "SELECT c.category_name, AVG(p.unit_price) AS avg_price FROM products p JOIN categories c ON p.category_id = c.category_id GROUP BY c.category_name;"},
    {"input": "List the top 5 customers by total order value.", "query": "SELECT c.customer_id, c.company_name, SUM(o.order_total) AS total_value FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.company_name ORDER BY total_value DESC LIMIT 5;"},
    {"input": "Find all products that are out of stock.", "query": "SELECT * FROM products WHERE units_in_stock = 0;"},
    {"input": "List all orders placed in the last 30 days.", "query": "SELECT * FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '30 days';"},
    {"input": "Calculate the total number of products in each category.", "query": "SELECT c.category_name, COUNT(*) AS product_count FROM products p JOIN categories c ON p.category_id = c.category_id GROUP BY c.category_name;"},
]

# Prompt template
EXAMPLE_PROMPT = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

EDA_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=POSTGRESQL_EXAMPLES,
    example_prompt=EXAMPLE_PROMPT,
    prefix="You are a PostgreSQL expert. Given an user input query, create a syntactically correct PostgreSQL query to run and retrieve all relevant raw data from the database which is required for the user query without performing aggregation like ""Count"" and ""GROUP BY"", return all the rows and columns(including date, month or year if it is required for the user query) required for the user query. Rather than only {top_k} rows return all the rows of the relevant data for the user query. \n\nHere is the relevant database info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)

QNA_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=POSTGRESQL_EXAMPLES,
    example_prompt=EXAMPLE_PROMPT,
    prefix="You are a PostgreSQL expert. Given an user input query, create a syntactically correct PostgreSQL query to run and retrieve all relevant raw data from the database which is required for the user query. Rather than only {top_k} rows return all the rows of the relevant data for the user query. \n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding PostgreSQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)