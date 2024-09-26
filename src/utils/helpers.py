import logging
from lida import Manager, TextGenerationConfig, llm
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_google_vertexai import ChatVertexAI
import pandas as pd
import matplotlib.pyplot as plt
import base64
from datetime import datetime
from PIL import Image
import re
import os
import io
from decimal import Decimal
import psycopg2
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
from src.utils.constants import VIZ_CONFIG


# Initialize LIDA Manager
def initialize_lida_manager():
    try:
        logger.info("Initializing LIDA Manager...")
        text_gen = llm("palm")
        lida = Manager(text_gen=text_gen)
        logger.info("LIDA Manager initialized successfully.")
        return lida, text_gen
    except Exception as e:
        logger.error(f"Error initializing LIDA Manager: {e}")
        raise


# Set up database connection
def setup_database_connection():
    try:
        db = SQLDatabase.from_uri(os.getenv("DATABASE_URI"), sample_rows_in_table_info=3)
        logger.info("Database connection setup successfully.")
        return db
    except Exception as e:
        logger.error(f"Error setting up database connection: {e}")
        raise


# Set up PostgreSQL database connection
def postgresql_database_connection():
    try:
        db = SQLDatabase.from_uri(os.getenv("POSTGRESQL_DATABASE_URI"), sample_rows_in_table_info=3)
        logger.info("Connected to PostgreSQL database successfully.")
        return db
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL database: {e}")
        raise


# Create SQL Chain
def create_sql_chain(db, prompt_template):
    try:
        logger.info("Creating SQL chain...")
        llm = initialize_llm()
        chain = create_sql_query_chain(llm, db, prompt_template)
        logger.info("SQL chain created successfully.")
        return chain
    except Exception as e:
        logger.error(f"Error creating SQL chain: {e}")
        raise


# Execute SQL Query
def execute_sql_query(query):
    try:
        logger.info(f"Executing SQL query: {query}")
        mydb = psycopg2.connect(database=os.getenv("DB_DATABASE"),
                                user=os.getenv("DB_USER"),
                                host=os.getenv("DB_HOST"),
                                password=os.getenv("DB_PASSWORD"),
                                port=os.getenv("DB_PORT"))
        cursor = mydb.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        cursor.close()
        mydb.close()
        logger.info("SQL query executed successfully.")
        return data, column_names
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        raise


# Create Pandas DataFrame
def create_dataframe(data, column_names):
    try:
        logger.info("Creating DataFrame from query result...")
        df = pd.DataFrame(data, columns=column_names)

        # Data type conversions
        for col in df.columns:
            if df[col].dtype == object:
                if len(df[col]) > 0 and isinstance(df[col].iloc[0], Decimal):
                    df[col] = df[col].astype(float)
                elif len(df[col]) > 0 and isinstance(df[col].iloc[0], datetime):
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = pd.to_numeric(df[col], errors='ignore')

        logger.info("DataFrame created successfully.")
        return df
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        raise


# Generate Visualization
def generate_visualization(lida, summary, user_query):
    try:
        logger.info("Generating visualization...")
        visualization = lida.visualize(summary=summary, goal=user_query, textgen_config=TextGenerationConfig(**VIZ_CONFIG))
        logger.info("Visualization generated successfully.")
        return visualization
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        raise


# Display Visualization
def display_visualization(charts):
    try:
        if not charts:
            logger.info("No charts to display.")
            return None

        logger.info(f"Displaying {len(charts) if isinstance(charts, list) else 1} charts...")
        n_charts = len(charts) if isinstance(charts, list) else 1
        n_cols = min(3, n_charts)
        n_rows = (n_charts - 1) // n_cols + 1

        fig = plt.figure(figsize=(10 * n_cols, 7 * n_rows))

        for i, chart in enumerate(charts if isinstance(charts, list) else [charts]):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            img_data = base64.b64decode(chart.raster)
            img = Image.open(io.BytesIO(img_data))
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(left=0.2)
        logger.info("Charts displayed successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error displaying visualization: {e}")
        raise


# Initialize LLM
def initialize_llm():
    try:
        logger.info("Initializing LLM...")
        llm = ChatVertexAI(model="gemini-1.5-pro")
        logger.info("LLM initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise


# Clean SQL Query
def clean_sql_query(query):
    try:
        logger.info("Cleaning SQL query...")
        query = re.sub(r'```sql|```', '', query)
        query = query.replace('`', '')
        query = query.strip()
        logger.info("SQL query cleaned.")
        return query
    except Exception as e:
        logger.error(f"Error cleaning SQL query: {e}")
        raise


# Create QnA Chain
def create_qna_chain(write_query, execute_query, answer):
    try:
        logger.info("Creating QnA chain...")
        chain = (
            RunnablePassthrough.assign(query=write_query)
            .assign(cleaned_query=lambda x: clean_sql_query(x['query']))
            .assign(result=lambda x: execute_query.run(x['cleaned_query']))
            | answer
        )
        logger.info("QnA chain created successfully.")
        return chain
    except Exception as e:
        logger.error(f"Error creating QnA chain: {e}")
        raise
