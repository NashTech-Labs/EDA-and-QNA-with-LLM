import os
import time
import psutil
import threading
from flask import Flask, request, session, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from sqlalchemy import create_engine, text
import google.generativeai as genai
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

load_dotenv()

# Load environment variables
DB_USER = os.getenv('DB_USER', 'dbuser')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'dbpassword')
DB_NAME = os.getenv('DB_DATABASE', 'financialdb')
DB_HOST = os.getenv('DB_HOST', 'mysql')  # 'mysql' is the service name in docker-compose
DB_PORT = os.getenv('DB_PORT', '3306')

# SQLAlchemy engine
DATABASE_URI = f"mysql+mysqlconnector://sid:Sid123@localhost/propertiesdb"
engine = create_engine(DATABASE_URI)

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.urandom(24)  # Generate a random secret key for sessions

# Configure Flask session
app.config['SESSION_TYPE'] = 'filesystem'  # Store sessions in filesystem
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # Session lifetime in seconds (30 minutes)


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400

    # Set session variables
    session['model_name'] = "gemini-1.5-pro"
    session['email'] = email
    session['chat_history'] = []  # Initialize empty chat history

    return jsonify({"message": "Registered successfully", "email": email})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400

    # Set session variables
    session['model_name'] = "gemini-1.5-pro"
    session['email'] = email
    if 'chat_history' not in session:
        session['chat_history'] = []

    return jsonify({"message": "Logged in successfully", "email": email})


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()  # Clear all session data
    return jsonify({"message": "Logged out successfully"})


@app.route('/api/get_tables', methods=['GET'])
def get_tables():
    if 'email' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        with engine.connect() as conn:
            query = text("SHOW TABLES;")
            result = conn.execute(query)
            tables = [row[0] for row in result]

            return jsonify({
                "message": "Tables retrieved successfully",
                "tables": tables
            })

    except SQLAlchemyError as e:
        return jsonify({"error": f"Failed to fetch tables: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to fetch tables: {str(e)}"}), 500


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv_and_update_db():
    try:
        if 'email' not in session:
            return jsonify({"error": "Not authenticated"}), 401

        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected for uploading."}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed."}), 400

        # Get the table name from the request
        table_name = request.form.get('table_name')
        if not table_name:
            return jsonify({"error": "Table name not provided."}), 400

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV file: {e}"}), 400

        with engine.connect() as conn:
            query = f"SELECT * FROM {table_name} LIMIT 1;"
            db_columns = pd.read_sql(query, conn).columns.tolist()

        db_columns = [col for col in db_columns if col != 'id']

        if set(df.columns) != set(db_columns):
            return jsonify({"error": "CSV columns do not match the database columns."}), 400

        try:
            df.to_sql(table_name, engine, if_exists='append', index=False)
        except Exception as e:
            return jsonify({"error": f"Failed to insert data into the table '{table_name}': {e}"}), 500

        return jsonify({"message": f"CSV data successfully inserted into the table '{table_name}'."}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process CSV file: {e}"}), 500


@app.route('/api/describe_table', methods=['POST'])
def describe_table():
    try:
        if 'email' not in session:
            return jsonify({"error": "Not authenticated"}), 401

        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected for uploading."}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed."}), 400

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV file: {e}"}), 400

        # Analyze and store column information
        column_info = []
        for column in df.columns:
            # Get sample non-null value for better type inference
            sample_value = df[column].dropna().iloc[0] if not df[column].empty else None

            info = {
                "name": ''.join(e for e in column if e.isalnum() or e == '_'),
                "original_dtype": str(df[column].dtype),
                "sample_value": str(sample_value),
                "has_nulls": bool(df[column].isnull().sum() > 0),
                "max_length": int(df[column].astype(str).str.len().max())
            }
            column_info.append(info)

        # Store the analysis in session
        analysis = {
            "column_info": column_info,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        session['table_analysis'] = analysis


        return jsonify({
            "message": "Table analysis completed successfully.",
            "analysis": analysis
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to analyze table: {e}"}), 500


@app.route('/api/create_table', methods=['POST'])
def create_table():
    try:
        if 'email' not in session:
            return jsonify({"error": "Not authenticated"}), 401

        if 'table_analysis' not in session:
            return jsonify({"error": "Please analyze a CSV file first using /api/create_table"}), 400

        data = request.json
        table_name = data.get('table_name')

        if not table_name:
            return jsonify({"error": "Table name is required"}), 400

        if not table_name.isalnum():
            return jsonify({"error": "Table name must contain only alphanumeric characters"}), 400

        analysis = session['table_analysis']

        def map_dtype_to_sql(col_info):
            dtype = col_info['original_dtype']
            sample_value = col_info['sample_value']
            max_length = col_info['max_length']

            if 'int' in dtype.lower():
                try:
                    if sample_value and -2147483648 <= float(sample_value) <= 2147483647:
                        return 'INT'
                    return 'BIGINT'
                except ValueError:
                    return 'INT'
            elif 'float' in dtype.lower():
                return 'DOUBLE'
            elif 'datetime' in dtype.lower():
                return 'DATETIME'
            elif 'bool' in dtype.lower():
                return 'BOOLEAN'
            else:
                if max_length <= 255:
                    return f'VARCHAR({max_length})'
                return 'TEXT'

        columns = []
        for col in analysis['column_info']:
            sql_type = map_dtype_to_sql(col)
            null_constraint = "" if col['has_nulls'] else "NOT NULL"
            columns.append(f"{col['name']} {sql_type} {null_constraint}")

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            {','.join(columns)},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            with engine.connect() as conn:
                conn.execute(text(create_table_query))
        except SQLAlchemyError as e:
            return jsonify({"error": f"Failed to create table: {str(e)}"}), 500

        return jsonify({
            "message": f"Table '{table_name}' created successfully",
            "create_table_query": create_table_query
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to create table: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5007)