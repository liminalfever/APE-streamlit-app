from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ape import APE, APEScore

load_dotenv()

MODEL_ID = "francescoortame/ape-bge-small"
ACTIVITY_THRESHOLD = 0.6
LOGO_PATH = Path(__file__).parent / "media" / "ape-transparent.png"


@st.cache_resource(show_spinner="Loading model")
def load_model() -> APE:
    # Cached across reruns *and* across user sessions: the model is loaded once per process.
    return APE.from_pretrained(MODEL_ID)


def slot_name(i: int) -> str:
    return f"slot {i}"


def active_slot_indices(slot_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Slots that `APE.encode` returns an embedding for, in slot order.

    Mirrors the selection rule in `APE.encode`: every slot whose activity score exceeds
    the threshold is kept, and if there is none, the most active slot is kept.
    """
    active = np.flatnonzero(slot_scores > threshold)
    if active.size == 0:
        active = np.array([int(slot_scores.argmax())])
    return active


def similarity_heatmap(
    sim: np.ndarray, row_labels: list[str], col_labels: list[str]
) -> alt.LayerChart:
    """Annotated heatmap of a [n_rows, n_cols] cosine similarity matrix."""
    df = pd.DataFrame(
        [
            {"row": r, "col": c, "similarity": float(sim[i, j])}
            for i, r in enumerate(row_labels)
            for j, c in enumerate(col_labels)
        ]
    )

    lo = min(0.0, float(sim.min()))
    mid = lo + 0.55 * (1.0 - lo)  # switch the annotation color on dark cells

    base = alt.Chart(df).encode(
        x=alt.X("col:N", sort=col_labels, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("row:N", sort=row_labels, title=None),
    )
    cells = base.mark_rect(cornerRadius=4, stroke="rgba(128,128,128,0.25)").encode(
        color=alt.Color(
            "similarity:Q",
            scale=alt.Scale(domain=[lo, 1.0], scheme="blues"),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("row:N", title="Row"),
            alt.Tooltip("col:N", title="Column"),
            alt.Tooltip("similarity:Q", title="Cosine", format=".3f"),
        ],
    )
    labels = base.mark_text(fontSize=15, fontWeight="bold").encode(
        text=alt.Text("similarity:Q", format=".2f"),
        color=alt.condition(
            alt.datum.similarity > mid, alt.value("white"), alt.value("#16407a")
        ),
    )
    return (cells + labels).properties(height=alt.Step(70))


model = load_model()

st.image(str(LOGO_PATH))

tab1, tab2 = st.tabs(["Single sentence", "Comparison"])

with tab1:
    st.text_input(
        "Sentence",
        value="Rome was built in 753 BC and is now the capital of Italy, while Paris is a big city indeed, Bangkok is crowded and New York is very busy.",
        key="sent",
    )

    if st.button("Encode", icon="🦧"):
        sentence = st.session_state.sent.strip()
        if not sentence:
            st.session_state.pop("single_result", None)
            st.warning("Please enter a sentence.")
        else:
            with st.spinner("Encoding"):
                prop_embds, act_scores = model.encode(
                    sentence,
                    activity_threshold=ACTIVITY_THRESHOLD,
                    return_scores=True,
                )
                slot_scores = model.activity_scores([sentence])[0]  # [num_slots]

            # Sanity check: the slots we derive from `slot_scores` (see `active_slot_indices`)
            # must be exactly the ones `encode` returned embeddings for.
            assert len(active_slot_indices(slot_scores, ACTIVITY_THRESHOLD)) == len(prop_embds), (
                "Active slots do not match the encoded propositions"
            )

            # Persist the results: widget interactions rerun the script, and the
            # button is only True on the run right after the click.
            st.session_state.single_result = {
                "sentence": sentence,
                "n_props": len(prop_embds),
                "prop_embds": prop_embds,  # [n_props, D], one row per active slot
                "act_scores": act_scores,
                "slot_scores": slot_scores,
            }

    result = st.session_state.get("single_result")
    if result is not None:
        col1, col2, col3 = st.columns([5,1,5])

        with col1:
            st.subheader("Activity scores")
            st.caption("These scores determine the predicted activity level of each proposition embedding.")

            n = result["n_props"]
            st.text(f"APE identified {n} active proposition{'' if n == 1 else 's'}.")

            # One row per slot, so that every slot gets its own labeled bar.
            slot_df = pd.DataFrame(
                {"activity": result["slot_scores"]},
                index=[slot_name(i) for i in range(len(result["slot_scores"]))],
            )
            st.bar_chart(slot_df, horizontal=True, sort=False, y_label="Predicted activity level")

        with col3:
            st.subheader("Proposition Seeding")
            st.caption("Interpret embeddings by choosing a seed proposition and comparing it to the APE representation.")
            seed = st.text_input("Seed Proposition", key="seed").strip()

            if seed:
                # Rows: the seed's embedding(s). Columns: the source sentence's active
                # proposition embeddings. Both come from the same encoder, so cosine
                # similarity is directly comparable.
                seed_embds = model.encode(seed, activity_threshold=ACTIVITY_THRESHOLD)
                sim = APEScore(seed_embds, result["prop_embds"]).similarity_matrix()

                n_seed = len(seed_embds)
                if n_seed > 1:
                    st.caption(
                        f"APE split this seed into {n_seed} propositions. "
                        "Seeds work best when they are a single atomic proposition."
                    )

                row_labels = ["Seed"] if n_seed == 1 else [f"Seed {i}" for i in range(n_seed)]
                active_slots = active_slot_indices(result["slot_scores"], ACTIVITY_THRESHOLD)
                col_labels = [slot_name(i) for i in active_slots]
                st.caption("Cosine similarity scores.")
                st.altair_chart(
                    similarity_heatmap(sim, row_labels, col_labels), width="stretch"
                )

with tab2:
    col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="bottom")

    with col1:
        st.text_input(
            "Sentence A",
            value="Rome was built in 753 BC and is now the capital of Italy.",
            key="sent_a",
        )

    with col2:
        st.text_input(
            "Sentence B",
            value="Paris is a beautiful city and Rome was built in 753 BC.",
            key="sent_b",
        )

    with col3:
        compare_clicked = st.button("Compare")

    if compare_clicked:
        sentence_a = st.session_state.sent_a.strip()
        sentence_b = st.session_state.sent_b.strip()
        if not sentence_a or not sentence_b:
            st.session_state.pop("comparison_result", None)
            st.warning("Please enter both sentences.")
        else:
            with st.spinner("Encoding"):
                prop_embds, act_scores = model.encode(
                    [sentence_a, sentence_b],
                    activity_threshold=ACTIVITY_THRESHOLD,
                    return_scores=True,
                )

            st.session_state.comparison_result = {
                "sentences": (sentence_a, sentence_b),
                "prop_embds": prop_embds,  # [emb_a, emb_b], each [n_props, D]
                "act_scores": act_scores,
            }

    # Read it back with st.session_state.get("comparison_result") when adding the
    # similarity matrix; it survives reruns until the next click on "Compare".