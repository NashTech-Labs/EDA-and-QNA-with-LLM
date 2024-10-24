# Data Exploration and Q&A with LLM

This project implements an interactive data exploration and question-answering system using Large Language Models (LLMs). It provides two main functionalities:
1. Exploratory Data Analysis (EDA) with automated visualization generation
2. Natural language question-answering over SQL databases

## Features

- **EDA Pipeline**
  - Automated SQL query generation from natural language
  - Data visualization using LIDA
  - Interactive chart editing capabilities
  - Visual data exploration

- **Q&A Pipeline**
  - Natural language to SQL conversion
  - Automated answer generation
  - Result visualization
  - Data presentation in tabular format

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Google Cloud Platform account (for Vertex AI)
- Required API keys:
  - Groq API
  - Google Cloud credentials

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file with the following:
```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_credentials.json
DATABASE_URI=your_database_uri
POSTGRESQL_DATABASE_URI=your_postgresql_uri
DB_DATABASE=your_database_name
DB_USER=your_database_user
DB_HOST=your_database_host
DB_PASSWORD=your_database_password
DB_PORT=your_database_port
```

## Project Structure

```
src/
├── pipeline/
│   ├── QNA_pipeline.py    # Question-answering pipeline implementation
│   └── eda_pipeline.py    # EDA pipeline implementation
└── utils/
    ├── constants.py       # Configuration and constant values
    └── helpers.py         # Utility functions and helpers
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application through your web browser at `http://localhost:8501`

3. Choose your analysis type:
   - **Perform EDA**: Enter a natural language query about what you want to explore in your data
   - **Ask Questions**: Enter a specific question about your data

4. View the results:
   - For EDA: Interactive visualizations with editing capabilities
   - For Q&A: Text answers with supporting data tables

## Features Details

### EDA Pipeline
- Converts natural language queries to SQL
- Automatically generates relevant visualizations
- Supports interactive chart editing
- Provides data summaries and insights

### Q&A Pipeline
- Natural language question processing
- SQL query generation and execution
- Answer generation with context
- Result visualization and presentation

## Error Handling

The application includes comprehensive error handling and logging:
- All operations are logged with appropriate detail levels
- User-friendly error messages
- Graceful failure handling
- Debug information in logs

## Acknowledgments

- LIDA for visualization generation
- Langchain for LLM integration
- Streamlit for the web interface
- Google Vertex AI for language model capabilities