import streamlit as st
from retriever import Retriever
import os

st.set_page_config(layout="wide")
st.title("🔍 Dataset Recommendation UI")


@st.cache_resource
def initialize_retreiver():
    print("Initializing retriever model...")
    retriever = Retriever()

    return retriever


retriever = initialize_retreiver()

# Create two columns
col1, col2 = st.columns(2)

# Column 1 for user input, number of relevant results, and search button
with col1:
    st.header("User Input")
    use_reranker = st.toggle("Use Re-ranker", value=True)
    if use_reranker:
        openai_key = st.text_input(
            "OpenAI API Key",
            key="openai_key",
            help="sk-***************************",
            type="password",
        )

    user_input = st.text_area("Enter search query", key="user_input")

    st.subheader("Number of Top Results to Retrieve")
    top_k_results = st.number_input(
        "Select top K results to retrieve",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        key="top_k",
    )

    # A button to perform the search
    search_button = st.button("Search")

avatar_list = ["🥇", "🥈", "🥉", "🤖", "🤖"]
# Column 2 for showing results inside cards
with col2:
    st.header("Recommeded Datasets:")

    if search_button:
        if use_reranker and openai_key == "":
            st.warning(
                "No OpenAI API key was found! If you don't want to use re-ranker, please switch off the toggle.",
                icon="⚠️",
            )
        elif user_input == "":
            st.warning("No user input!", icon="⚠️")
        else:
            if use_reranker:
                os.environ["OPENAI_API_KEY"] = openai_key
            candidates = retriever.retrieve_candidates(
                user_input, use_reranker=use_reranker, top_k=top_k_results
            )

            print(len(candidates))
            # # Placeholder for actual search logic
            # # This should be replaced with actual search results
            for i, candidate in enumerate(candidates):
                candidate = candidate.to_dict(orient="records")[0]

                domain = (
                    candidate["for_tasks"]
                    .replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                )

                with st.chat_message("user", avatar=avatar_list[i]):
                    st.markdown(f"**Dataset Name:** {candidate['name']}")
                    st.markdown(f"**Topics:** {domain}")
                    st.markdown(f"**Modality:** {candidate['data_type']}")
                    # st.markdown(f"**Published Year:** {int(candidate['published_year'])}")
                    st.markdown(
                        f"**Cited by:** {len(candidate['paper_lists']):,} papers"
                    )
                    st.markdown(f"**Source:** {candidate['url']}")
                    st.markdown("---")
                    st.markdown(f"**Description:** {candidate['description']}")
