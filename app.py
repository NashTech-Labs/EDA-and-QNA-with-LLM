import streamlit as st
import logging
from src.pipeline.eda_pipeline import run_eda_pipeline, edit_chart
from src.pipeline.QNA_pipeline import run_qna_pipeline
from src.utils.helpers import display_visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Data Exploration and Q&A with LLM")

# Custom CSS to improve the look
CUSTOM_CSS = """
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    .stSelectbox [data-baseweb="select"] {
        margin-top: 10px;
    }
</style>
"""

def initialize_session_state():
    """Initialize session state variables."""
    try:
        for key in ['result_data', 'df', 'fig', 'last_option', 'qna_answer', 'chart', 'summary', 'lida', 'edit_mode']:
            if key not in st.session_state:
                st.session_state[key] = None
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}", exc_info=True)
        st.error("An error occurred while initializing the application. Please try reloading the page.")

def run_pipeline(option, user_query):
    """Run the selected pipeline based on user input."""
    try:
        if option == "Perform EDA":
            st.session_state['df'], st.session_state['fig'], st.session_state['chart'], \
            st.session_state['summary'], st.session_state['lida'] = run_eda_pipeline(user_query)
        else:  # Ask Questions
            answer, df = run_qna_pipeline(user_query)
            st.session_state['qna_answer'] = answer
            st.session_state['df'] = df
        st.success(f"{option} pipeline completed!")
    except Exception as e:
        logger.error(f"Error in {option} pipeline: {str(e)}", exc_info=True)
        st.error(f"An error occurred while running the {option} pipeline. Please check the logs for details.")

def handle_chart_editing():
    """Handle chart editing functionality."""
    try:
        if st.button("Edit Chart" if not st.session_state['edit_mode'] else "Cancel Edit"):
            st.session_state['edit_mode'] = not st.session_state['edit_mode']

        if st.session_state['edit_mode']:
            instructions = st.text_area("Enter editing instructions (comma-separated):", height=100)
            if st.button("Apply Edits") and instructions:
                with st.spinner("Editing chart..."):
                    instructions_list = [instr.strip() for instr in instructions.split(',')]
                    edited_chart = edit_chart(st.session_state['lida'], st.session_state['chart'].code,
                                              st.session_state['summary'], instructions_list)
                    st.session_state['fig'] = display_visualization(edited_chart)
                    st.success("Chart edited successfully!")
                    st.pyplot(st.session_state['fig'])
                    st.session_state['edit_mode'] = False
            elif not instructions:
                st.warning("Please enter editing instructions before applying edits.")
    except Exception as e:
        logger.error(f"Error handling chart editing: {str(e)}", exc_info=True)
        st.error("An error occurred while editing the chart. Please try again.")

def main():
    try:
        st.title("Data Exploration and Q&A with LLM")
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.markdown('<p class="big-font">This app allows you to either perform Exploratory Data Analysis (EDA) or ask questions about your data using an LLM-based pipeline.</p>', unsafe_allow_html=True)

        initialize_session_state()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Choose Your Analysis")
            option = st.selectbox("Select an option:", ("Select an option", "Perform EDA", "Ask Questions"))

            if option != st.session_state['last_option']:
                for key in ['fig', 'df', 'result_data', 'qna_answer', 'chart', 'summary', 'lida', 'edit_mode']:
                    st.session_state[key] = None
                st.session_state['last_option'] = option

            if option != "Select an option":
                user_query = st.text_area("Enter your query:", height=100)

                if user_query and st.button(f"Run {option} Pipeline", key="run_pipeline"):
                    with st.spinner(f"Running {option} pipeline..."):
                        run_pipeline(option, user_query)

        with col2:
            if option == "Perform EDA" and st.session_state['fig'] is not None:
                st.pyplot(st.session_state['fig'])
                handle_chart_editing()
            elif option == "Ask Questions" and st.session_state['qna_answer'] is not None:
                st.info(f"Answer: {st.session_state['qna_answer']}")

            if st.session_state['df'] is not None:
                with st.expander("Show Data", expanded=False):
                    st.dataframe(st.session_state['df'].head(10))

    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please check the logs for details or try reloading the page.")

if __name__ == "__main__":
    main()