import logging
import os
import random
from json import JSONDecodeError
from pathlib import Path
from typing import List

import streamlit as st
from annotated_text import annotation
from markdown import markdown
from ui.utils import haystack_is_ready, query, send_feedback, get_backlink, Query, read_query_file, haystack_version, \
    DENSE, SPARSE
from spacy.lang.en import English

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "")

# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", "50"))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", "5"))

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", str(Path(__file__).parent / "eval_labels_example.csv"))

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))

QUERY_EXAMPLES: List[Query] = read_query_file("data/Dev_queries.json")
QUERY_EXAMPLES.extend(read_query_file("data/Test_queries.json"))


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():
    nlp = English()
    tokenizer = nlp.tokenizer

    st.set_page_config(page_title="CoreSearch Demo", page_icon="img/intel_logo.png")

    # Persistent state
    set_state_if_absent("passage", None)
    set_state_if_absent("question", random.choice(QUERY_EXAMPLES))
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None
        st.session_state.passage = None

    # Title
    st.write("# CoreSearch Demo")
    st.write(markdown('This Demo illustrate an noval event<sup>\*</sup> coreference<sup>\*\*</sup> search application'
                      ' in a very large document collection. The input to the model is a paragraph where the event '
                      'of interest is highlighted.'), unsafe_allow_html=True)
    st.write(markdown('<sup>\*</sup>Event is a word or phrase which denote an activity. Events occur in a specific time and place, '
             'and usually involve participants (event examples: "earthquake", "accident", "game", "conference", '
             '"shooting", etc...)'), unsafe_allow_html=True)
    st.write(markdown(
        '<sup>\*\*</sup>Coreferring relation between events, indicate if two mentions of an event '
        'refers to the same underline event. (for examples, the event in the sentence "In April 2010, an **_earthquake_** originated in the Yushu Tibetan"'
        ' and the event in the sentence "after the 2010 Yushu **tremor** destroyed the old school..", refers to the same earthquake event '
        'and therefor they corefere. However the earthquake event in the sentence - "The 2010 Chile **earthquake** and '
        'tsunami occurred off the coast of central Chile on Saturday" referes to a different event, and therefor do not corefere'), unsafe_allow_html=True)

    st.write("## Search Wikipedia Events ")

    hs_ver = haystack_version()

    # Sidebar
    st.sidebar.header("Options")
    select_ret = st.sidebar.radio(label="Retriever model for selecting top documents", options=[DENSE, SPARSE], index=1, on_change=reset_results)
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=5,
        max_value=200,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=5,
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

    side_text = st.sidebar.text_area("Copy here a passage from Wikipedia to search for:", value=" ", height=400, on_change=reset_results)
    copy_pressed = st.sidebar.button("Copy")
    # eval_mode = st.sidebar.checkbox("Evaluation mode")
    # debug = st.sidebar.checkbox("Show debug info")

    # Load queries

    # Search bar
    query_obj = st.session_state.question
    # query_ment = " ".join(query_obj.mention)
    # question = st.text_area("", value=query_text, on_change=reset_results)
    st.write(f"**_Search Query:_**")

    # HtmlFile = open("index.html", 'r', encoding='utf-8')
    # source_code = HtmlFile.read()
    # st.components.v1.html(html=source_code, width=None, height=None, scrolling=True)
    # text_area = st.text_area("", value=" ".join(query_obj.context), height=400, on_change=reset_results)

    if copy_pressed and side_text:
        query_obj = Query()
        query_obj.context = [tok.text for tok in tokenizer(side_text.strip())]
        query_obj.startIndex = 0
        query_obj.endIndex = 0
        query_obj.mention = [query_obj.context[0]]
        query_obj.id = -1
        query_obj.goldChain = -1
        st.session_state.question = query_obj

    col11, col12 = st.columns(2)
    start_index_val = int(col11.text_input("Mention Start Index", value=str(query_obj.startIndex), key="startIndex"))
    end_index_val = int(col12.text_input("Mention End Index", value=str(query_obj.endIndex), key="endIndex"))

    query_obj.startIndex = start_index_val
    query_obj.endIndex = end_index_val
    query_obj.mention = query_obj.context[start_index_val:end_index_val+1]

    query_ment = " ".join(query_obj.context[start_index_val:end_index_val + 1])
    context = [f"{word}<span style=\"font-size: .5rem\">({str(i)})</span>" for i, word in enumerate(query_obj.context)]
    st.write(
        markdown(" ".join(context[:start_index_val]) +
                 str(annotation(query_ment, "MENT", "#808080")) + " ".join(context[end_index_val + 1:])),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")

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
            f"🧠 &nbsp;&nbsp; Performing neural search on {str(hs_ver['total_docs'])} documents..."
        ):
            try:
                st.session_state.results, st.session_state.raw_json = query(
                    query_obj, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever, retriever_model=select_ret
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
                st.markdown(f"**Wikipedia Source Article:** {source} -- **Relevance:** {result['relevance']} -  ")

            else:
                st.info(
                    "🤔 &nbsp;&nbsp; No answer found"
                )
                st.write("**Relevance:** ", result["relevance"])

            st.write("___")


main()
