import logging
from src.utils.constants import TEXT_GEN_CONFIG, EDA_PROMPT_TEMPLATE
from src.utils.helpers import (
    initialize_lida_manager,
    postgresql_database_connection,
    create_sql_chain,
    execute_sql_query,
    create_dataframe,
    generate_visualization,
    display_visualization,
    clean_sql_query
)

logger = logging.getLogger(__name__)

class EDAError(Exception):
    """Custom exception for EDA pipeline errors."""
    pass

def run_eda_pipeline(user_query):
    """
    Run the Exploratory Data Analysis (EDA) pipeline.

    Args:
        user_query (str): The user's query for EDA.

    Returns:
        tuple: A tuple containing DataFrame, figure, chart, summary, and LIDA instance.

    Raises:
        EDAError: If an error occurs during the EDA pipeline execution.
    """
    try:
        # Initialize LIDA manager
        lida, _ = initialize_lida_manager()

        # Setup database connection
        db = postgresql_database_connection()

        # Create SQL query chain
        chain = create_sql_chain(db, EDA_PROMPT_TEMPLATE)

        # Generate database context
        db_context = db.get_context()

        # Generate and execute SQL query
        query = chain.invoke({"question": user_query, "top_k": 3, "table_info": db_context})
        cleaned_query = clean_sql_query(query)
        logger.info(f"Cleaned SQL query: {cleaned_query}")

        data, column_names = execute_sql_query(cleaned_query)

        # Create DataFrame
        df = create_dataframe(data, column_names)
        logger.info(f"DataFrame created with shape: {df.shape}")

        # Generate visualization
        summary = lida.summarize(df, summary_method="default", textgen_config=TEXT_GEN_CONFIG)
        charts = generate_visualization(lida, summary, user_query)

        # Get the figure from display_visualization
        fig = display_visualization(charts[0])

        return df, fig, charts[0], summary, lida

    except Exception as e:
        logger.error(f"Error in EDA pipeline: {str(e)}", exc_info=True)
        raise EDAError(f"An error occurred during the EDA pipeline execution: {str(e)}") from e

def edit_chart(lida, code, summary, instructions, library="seaborn"):
    """
    Edit a chart based on given instructions.

    Args:
        lida: LIDA instance
        code (str): Original chart code
        summary (dict): Summary of the data
        instructions (list): List of editing instructions
        library (str): Visualization library to use

    Returns:
        object: Edited chart

    Raises:
        EDAError: If an error occurs during chart editing.
    """
    try:
        textgen_config = TEXT_GEN_CONFIG._replace(n=1, temperature=0, use_cache=True)
        edited_charts = lida.edit(code=code, summary=summary, instructions=instructions, library=library, textgen_config=textgen_config)
        return edited_charts[0]
    except Exception as e:
        logger.error(f"Error editing chart: {str(e)}", exc_info=True)
        raise EDAError(f"An error occurred while editing the chart: {str(e)}") from e