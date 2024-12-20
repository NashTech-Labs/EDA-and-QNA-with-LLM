import logging
from src.utils.constants import QNA_PROMPT_TEMPLATE, answer_prompt, EDA_PROMPT_TEMPLATE
from src.utils.helpers import (
    setup_database_connection,
    postgresql_database_connection,
    create_sql_chain,
    initialize_llm,
    create_qna_chain,
    execute_sql_query_for_ans,
    execute_sql_query_for_data,
    create_dataframe,
    clean_sql_query
)
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class QNAError(Exception):
    """Custom exception for QNA pipeline errors."""
    pass

def run_qna_pipeline(user_query):
    """
    Run the Question and Answer (QNA) pipeline.

    Args:
        user_query (str): The user's question.

    Returns:
        tuple: A tuple containing the answer and the corresponding DataFrame.

    Raises:
        QNAError: If an error occurs during the QNA pipeline execution.
    """
    try:
        # Setup database connection
        db = setup_database_connection()

        # Create SQL query chain
        execute_query = QuerySQLDataBaseTool(db=db)
        write_query = create_sql_chain(db, QNA_PROMPT_TEMPLATE)

        # # Generate database context
        db_context = db.get_context()
        #
        # # Save the context to a text file
        # with open('db_context.txt', 'w') as file:
        #     file.write(str(db_context))

        # with open('db_context.txt', 'r') as file:
        #     db_context = file.read()
        # Generate SQL query
        query = write_query.invoke({"question": user_query, "top_k": 3, "table_info": db_context})

        # Define answer generation chain
        answer = answer_prompt | initialize_llm() | StrOutputParser()
        chain = create_qna_chain(write_query, execute_sql_query_for_ans, answer)

        # Run the chain and get the answer
        result_data = chain.invoke({"question": user_query})
        logger.info(f"Generated answer: {result_data}")

        # Clean and execute the SQL query
        cleaned_query = clean_sql_query(query) if isinstance(query, str) else query.get('cleaned_query')
        logger.info(f"Cleaned SQL query: {cleaned_query}")

        data, column_names = execute_sql_query_for_data(cleaned_query)

        # Create DataFrame from the SQL results
        df = create_dataframe(data, column_names)
        logger.info(f"DataFrame created with shape: {df.shape}")

        return result_data, df

    except Exception as e:
        logger.error(f"Error in QNA pipeline: {str(e)}", exc_info=True)
        raise QNAError(f"An error occurred during the QNA pipeline execution: {str(e)}") from e