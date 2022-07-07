import os
import random
import sys
import logging
from pathlib import Path
from json import JSONDecodeError
from typing import List

import streamlit as st
from annotated_text import annotation
from markdown import markdown

from ui.utils import haystack_is_ready, query, send_feedback, upload_doc, haystack_version, get_backlink, Query, read_query_file


# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "")

# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", "5"))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", "3"))

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", str(Path(__file__).parent / "eval_labels_example.csv"))

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))

QUERY_EXAMPLES: List[Query] = read_query_file("data/Dev_queries.json")


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():

    st.set_page_config(page_title="Haystack Demo", page_icon="https://haystack.deepset.ai/img/HaystackIcon.png")

    # Persistent state
    set_state_if_absent("question", QUERY_EXAMPLES[0])
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    st.write("# CoreSearch Demo")
    st.write("## Search Wikipedia Events ")

    # Sidebar
    st.sidebar.header("Options")
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=5,
        max_value=50,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results,
    )
    top_k_reader = st.sidebar.slider(
        "Max. number of answers",
        min_value=1,
        max_value=10,
        value=DEFAULT_NUMBER_OF_ANSWERS,
        step=1,
        on_change=reset_results,
    )
    eval_mode = st.sidebar.checkbox("Evaluation mode")
    debug = st.sidebar.checkbox("Show debug info")

    # Load queries

    # Search bar
    query_obj = st.session_state.question
    query_ment = " ".join(query_obj.mention)
    # question = st.text_area("", value=query_text, on_change=reset_results)
    st.write(f"**_Search Query (Coreference ID={str(query_obj.goldChain)}):_**")
    st.write(
        markdown(" ".join(query_obj.context[:query_obj.startIndex]) +
                 str(annotation(query_ment, "MENT", "#808080")) + " ".join(query_obj.context[query_obj.endIndex + 1:])),
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")

    query_text = " ".join(query_obj.context)
    # Get next random question from the CSV
    if col2.button("Random question"):
        reset_results()
        # new_row = df.sample(1)
        st.session_state.question = random.choice(QUERY_EXAMPLES)
        # st.session_state.answer = new_row["Answer"].values[0]
        st.session_state.random_question_requested = True
        # Re-runs the script setting the random question as the textbox value
        # Unfortunately necessary as the Random Question button is _below_ the textbox
        # raise st.scriptrunner.script_runner.RerunException(st.script_request_queue.RerunData(None))
        st.experimental_rerun()
    st.session_state.random_question_requested = False

    run_query = run_pressed and not st.session_state.random_question_requested

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Starting..."):
        if not haystack_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error.")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and query_obj:
        reset_results()
        # st.session_state.question = question

        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on documents..."
        ):
            try:
                st.session_state.results, st.session_state.raw_json = query(
                    st.session_state.question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever
                )
            except JSONDecodeError as je:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    if st.session_state.results:

        # Show the gold answer if we use a question of the given set
        if eval_mode and st.session_state.answer:
            st.write("## Correct answer:")
            st.write(st.session_state.answer)

        st.write("## Results:")

        for count, result in enumerate(st.session_state.results):
            if result["answer"]:
                answer, context = result["answer"], result["context"]
                start_idx = context.find(answer)
                end_idx = start_idx + len(answer)
                # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
                st.write(
                    markdown(context[:start_idx] + str(annotation(answer, "ANSWER", "#6a0dad")) + context[end_idx:]),
                    unsafe_allow_html=True,
                )
                source = ""
                url, title = get_backlink(result)
                if url and title:
                    source = f"[{result['document']['meta']['title']}]({result['document']['meta']['url']})"
                else:
                    source = f"{result['source']}"
                st.markdown(f"**Relevance:** {result['relevance']} -  **Source:** {source}")

            else:
                st.info(
                    "🤔 &nbsp;&nbsp; No answer found"
                )
                st.write("**Relevance:** ", result["relevance"])

            if eval_mode and result["answer"]:
                # Define columns for buttons
                is_correct_answer = None
                is_correct_document = None

                button_col1, button_col2, button_col3, _ = st.columns([1, 1, 1, 6])
                if button_col1.button("👍", key=f"{result['context']}{count}1", help="Correct answer"):
                    is_correct_answer = True
                    is_correct_document = True

                if button_col2.button("👎", key=f"{result['context']}{count}2", help="Wrong answer and wrong passage"):
                    is_correct_answer = False
                    is_correct_document = False

                if button_col3.button(
                    "👎👍", key=f"{result['context']}{count}3", help="Wrong answer, but correct passage"
                ):
                    is_correct_answer = False
                    is_correct_document = True

                if is_correct_answer is not None and is_correct_document is not None:
                    try:
                        send_feedback(
                            query=question,
                            answer_obj=result["_raw"],
                            is_correct_answer=is_correct_answer,
                            is_correct_document=is_correct_document,
                            document=result["document"],
                        )
                        st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")
                    except Exception as e:
                        logging.exception(e)
                        st.error("🐞 &nbsp;&nbsp; An error occurred while submitting your feedback!")

            st.write("___")

        if debug:
            st.subheader("REST API JSON response")
            st.write(st.session_state.raw_json)


main()
