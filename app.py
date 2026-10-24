import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ape import APE, APEScore

load_dotenv()

MODEL_ID = "sapienzanlp/ape-bge-small"
ACTIVITY_THRESHOLD = 0.6
IMAGE_PATH = Path(__file__).parent / "media" / "ape.png"
LOGO_PATH = Path(__file__).parent / "media" / "sapienzanlp-transparent.png"

# Selectbox label -> (`APEScore` rule name, short description)
SCORING_RULES = {
    "Hungarian": (
        "hungarian",
        "Optimal one-to-one matching between the propositions of the two sentences, "
        "averaged over the matched pairs. Symmetric.",
    ),
    "MaxSim": (
        "maxsim",
        "Mean of each proposition's best match, averaged over both directions. Symmetric.",
    ),
    "Hypothesis-to-Premise": (
        "h_min_to_p",
        "Worst best-match of any hypothesis proposition among the premise propositions. "
        "Not symmetric: the result depends on which sentence is the hypothesis.",
    ),
    "Pooling": (
        "pool",
        "Cosine similarity of the mean-pooled proposition embeddings. Symmetric.",
    ),
}


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
    sim: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    row_title: str | None = None,
    col_title: str | None = None,
    max_plot_width: int = 560,
    highlight_rows: np.ndarray | None = None,
    highlight_cols: np.ndarray | None = None,
    dim_alpha: float = 0.2,
    partial_alpha: float = 0.5,
) -> alt.LayerChart:
    """Annotated heatmap of a [n_rows, n_cols] cosine similarity matrix.

    Cells are square and have a fixed size (they do not stretch with the page): at most
    100 px, shrinking only when many columns would not fit in `max_plot_width` pixels.

    `highlight_rows` / `highlight_cols` are indices of the rows / columns to emphasize:

    - Only rows (or only columns) given: those stay fully visible, everything else is
      drawn with opacity `dim_alpha`.
    - Both given: the cells where a highlighted row meets a highlighted column are fully
      visible and outlined in white, the rest of those rows and columns is drawn with
      opacity `partial_alpha`, and everything else with `dim_alpha`.

    Axis labels of rows / columns that are not highlighted are drawn with `dim_alpha`.
    """
    n_rows, n_cols = len(row_labels), len(col_labels)

    def hit_mask(n: int, highlighted: np.ndarray | None) -> np.ndarray | None:
        return None if highlighted is None else np.isin(np.arange(n), highlighted)

    row_hit = hit_mask(n_rows, highlight_rows)
    col_hit = hit_mask(n_cols, highlight_cols)
    crossing = row_hit is not None and col_hit is not None

    def cell_alpha(i: int, j: int) -> float:
        if crossing:
            if row_hit[i] and col_hit[j]:
                return 1.0
            return partial_alpha if (row_hit[i] or col_hit[j]) else dim_alpha
        if row_hit is not None:
            return 1.0 if row_hit[i] else dim_alpha
        if col_hit is not None:
            return 1.0 if col_hit[j] else dim_alpha
        return 1.0

    cell = max(32, min(100, max_plot_width // n_cols))
    lo = min(0.0, float(sim.min()))
    mid = lo + 0.55 * (1.0 - lo)  # switch the annotation color on dark cells

    records = []
    for i, r in enumerate(row_labels):
        for j, c in enumerate(col_labels):
            alpha = cell_alpha(i, j)
            records.append(
                {
                    "row": r,
                    "col": c,
                    "similarity": float(sim[i, j]),
                    "opacity": alpha,
                    # Dimmed cells are pale, so their annotation stays dark.
                    "white_text": bool(sim[i, j] > mid and alpha > dim_alpha),
                    "crossing": bool(crossing and row_hit[i] and col_hit[j]),
                }
            )
    df = pd.DataFrame(records)

    def label_opacity(labels: list[str], hit: np.ndarray | None):
        """Axis labels of rows / columns that are not highlighted are dimmed."""
        if hit is None:
            return alt.Undefined
        keep = [lab for lab, h in zip(labels, hit) if h]
        return {
            "condition": {
                "test": f"indexof({json.dumps(keep)}, datum.value) >= 0",
                "value": 1.0,
            },
            "value": dim_alpha,
        }

    base = alt.Chart(df).encode(
        x=alt.X(
            "col:N",
            sort=col_labels,
            title=col_title,
            axis=alt.Axis(labelAngle=0, labelOpacity=label_opacity(col_labels, col_hit)),
        ),
        y=alt.Y(
            "row:N",
            sort=row_labels,
            title=row_title,
            axis=alt.Axis(labelOpacity=label_opacity(row_labels, row_hit)),
        ),
        opacity=alt.Opacity("opacity:Q", scale=None, legend=None),  # use the values as-is
    )
    cells = base.mark_rect(cornerRadius=4, stroke="rgba(128,128,128,0.25)").encode(
        color=alt.Color(
            "similarity:Q",
            scale=alt.Scale(domain=[lo, 1.0], scheme="greens"),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("row:N", title=row_title or "Row"),
            alt.Tooltip("col:N", title=col_title or "Column"),
            alt.Tooltip("similarity:Q", title="Cosine", format=".3f"),
        ],
    )
    labels = base.mark_text(fontSize=15, fontWeight="bold").encode(
        text=alt.Text("similarity:Q", format=".2f"),
        color=alt.condition("datum.white_text", alt.value("white"), alt.value("#00441b")),
    )
    layers = [cells, labels]

    # White outline on the cells where a highlighted row meets a highlighted column.
    if crossing:
        outline = (
            alt.Chart(df[df["crossing"]])
            .mark_rect(fill="transparent", stroke="white", strokeWidth=3, cornerRadius=4)
            .encode(
                x=alt.X("col:N", sort=col_labels),
                y=alt.Y("row:N", sort=row_labels),
            )
        )
        layers.append(outline)

    return alt.layer(*layers).properties(width=alt.Step(cell), height=alt.Step(cell))


def show_heatmap(chart: alt.LayerChart) -> None:
    """Show a fixed-size heatmap centered in the current column."""
    with st.container(horizontal_alignment="center"):
        st.altair_chart(chart, width="content")


def n_active_text(n: int) -> str:
    return f"APE identified {n} active proposition{'' if n == 1 else 's'}."


def activity_bar_chart(slot_scores: np.ndarray) -> None:
    """One horizontal bar per slot, labeled `slot 0`, `slot 1`, ... in slot order."""
    slot_df = pd.DataFrame(
        {"activity": slot_scores},
        index=[slot_name(i) for i in range(len(slot_scores))],
    )
    st.bar_chart(slot_df, horizontal=True, sort=False, y_label="Predicted activity level")


def matched_indices(seed_embds, prop_embds) -> np.ndarray:
    """Index of the proposition that each seed embedding matches best (highest cosine)."""
    sim = APEScore(seed_embds, prop_embds).similarity_matrix()  # [n_seed, n_props]
    return np.unique(sim.argmax(axis=1))


def apply_comparison_seeds() -> None:
    """Encode the seeds typed in the Comparison tab and keep them in session state.

    Used as the button's `on_click`: callbacks run *before* the script reruns, so the
    similarity matrix (drawn above the button) already sees the new seeds. An empty
    box means "no seed" for that sentence, so clicking with both empty clears the highlight.
    """
    model = load_model()
    seeds = {}
    for side in ("a", "b"):
        text = st.session_state.get(f"seed_{side}", "").strip()
        seeds[side] = (
            model.encode(text, activity_threshold=ACTIVITY_THRESHOLD) if text else None
        )
    st.session_state.comparison_seeds = seeds  # {"a": [k, D] or None, "b": [k, D] or None}


model = load_model()

st.logo(str(LOGO_PATH), size="large", link="http://nlp.uniroma1.it/")

st.html("""
  <style>
    [alt=Logo] {
      height: 6rem;
      transform: translateY(2rem);  /* positive = down, tune to taste */
    }
  </style>
""")

st.image(str(IMAGE_PATH))

tab1, tab2 = st.tabs(["Single sentence", "Comparison"])

with tab1:
    col1, col2 = st.columns([4,1], vertical_alignment="center")

    with col1:
        st.text_area(
            "Sentence",
            value="Rome was built in 753 BC and is now the capital of Italy, while Paris is a big city indeed, Bangkok is crowded and New York is very busy.",
            key="sent",
            height=68
        )

    with col2:
        if st.button("Encode", icon="🦧", use_container_width=True):
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
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Activity Scores")
            st.caption("These scores determine the predicted activity level of each proposition embedding.")

            st.text(n_active_text(result["n_props"]))
            activity_bar_chart(result["slot_scores"])

        with col2:
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
                show_heatmap(
                    similarity_heatmap(sim, row_labels, col_labels, max_plot_width=260)
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
        compare_clicked = st.button("Compare", icon="🦧", use_container_width=True)

    if compare_clicked:
        sentence_a = st.session_state.sent_a.strip()
        sentence_b = st.session_state.sent_b.strip()
        if not sentence_a or not sentence_b:
            st.session_state.pop("comparison_result", None)
            st.session_state.pop("comparison_seeds", None)
            st.warning("Please enter both sentences.")
        else:
            with st.spinner("Encoding"):
                prop_embds, act_scores = model.encode(
                    [sentence_a, sentence_b],
                    activity_threshold=ACTIVITY_THRESHOLD,
                    return_scores=True,
                )
                slot_scores = model.activity_scores([sentence_a, sentence_b])  # [2, num_slots]

            # Same sanity check as in the single-sentence tab, for each sentence.
            for scores, embds in zip(slot_scores, prop_embds):
                assert len(active_slot_indices(scores, ACTIVITY_THRESHOLD)) == len(embds), (
                    "Active slots do not match the encoded propositions"
                )

            st.session_state.comparison_result = {
                "sentences": (sentence_a, sentence_b),
                "prop_embds": prop_embds,  # [emb_a, emb_b], each [n_props, D]
                "act_scores": act_scores,
                "slot_scores": slot_scores,  # row 0: sentence A, row 1: sentence B
            }

            # New sentences make the old seeds meaningless: clear them and their input boxes.
            # (Setting widget state is fine here, the boxes are only created further down.)
            st.session_state.pop("comparison_seeds", None)
            st.session_state.seed_a = ""
            st.session_state.seed_b = ""

    comparison = st.session_state.get("comparison_result")
    # A result saved by an earlier version of the app has no "slot_scores": ignore it
    # until the next click on "Compare".
    if comparison is not None and "slot_scores" in comparison:
        with st.expander("Activity scores"):
            st.subheader("Activity Scores")
            st.caption("These scores determine the predicted activity level for each proposition embedding.")

            col1, col2 = st.columns(2, vertical_alignment="bottom")
            for col, name, embds, scores in zip(
                (col1, col2),
                ("Sentence A", "Sentence B"),
                comparison["prop_embds"],
                comparison["slot_scores"],
            ):
                with col:
                    st.text(f"{name}: {n_active_text(len(embds))}")
                    activity_bar_chart(scores)

        with st.expander("Similarity matrix", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Proposition Similarity")
                st.caption("Cosine similarity between the proposition embeddings of the two sentences.")

                # Rows: sentence A's active propositions. Columns: sentence B's. Each one is
                # labeled with the slot it comes from, so it can be matched to the charts above.
                emb_a, emb_b = comparison["prop_embds"]
                slots_a, slots_b = (
                    active_slot_indices(scores, ACTIVITY_THRESHOLD)
                    for scores in comparison["slot_scores"]
                )
                scorer = APEScore(emb_a, emb_b)  # rows: A, columns: B (A = hypothesis if asymmetric)
                sim = scorer.similarity_matrix()

                # Proposition seeding: a seed for sentence A picks the row it matches best, a
                # seed for B the column. See `similarity_heatmap` for how they are highlighted.
                seeds = st.session_state.get("comparison_seeds") or {}
                seed_a, seed_b = seeds.get("a"), seeds.get("b")
                matched_rows = None if seed_a is None else matched_indices(seed_a, emb_a)
                matched_cols = None if seed_b is None else matched_indices(seed_b, emb_b)

                show_heatmap(
                    similarity_heatmap(
                        sim,
                        row_labels=[slot_name(i) for i in slots_a],
                        col_labels=[slot_name(i) for i in slots_b],
                        row_title="Sentence A",
                        col_title="Sentence B",
                        max_plot_width=320,  # must fit the 2/3-width column, axes included
                        highlight_rows=matched_rows,
                        highlight_cols=matched_cols,
                    )
                )

            with col2:
                st.subheader("Scoring")
                st.caption("Collapse the similarity matrix to a single score with scoring rules.")
                rule_label = st.selectbox("Scoring Rule", options=list(SCORING_RULES))
                rule, rule_description = SCORING_RULES[rule_label]

                rule_scorer = scorer
                if rule == "h_min_to_p":
                    # Not symmetric: let the user choose which sentence is the hypothesis.
                    hypothesis = st.segmented_control(
                        "Hypothesis",
                        options=["Sentence A", "Sentence B"],
                        default="Sentence A",
                        required=True,  # always keep one selected
                        key="hypothesis",
                        width="stretch",
                    )
                    if hypothesis == "Sentence B":
                        rule_scorer = APEScore(emb_b, emb_a)  # (hypothesis, premise)

                st.metric(
                    "Score",
                    f"{rule_scorer.score(rule):.3f}",
                    help=rule_description,
                    border=True,
                )
            
            st.subheader("Proposition Seeding")
            st.caption("Interpret embeddings by choosing a seed proposition and comparing it to the APE representation.")
            col1, col2, col3 = st.columns([2,2,1], vertical_alignment="bottom")
            with col1:
                st.text_input("Seed proposition for Sentence A", key="seed_a")
            with col2:
                st.text_input("Seed proposition for Sentence B", key="seed_b")
            with col3:
                st.button(
                    "Seed",
                    icon="🦧",
                    use_container_width=True,
                    on_click=apply_comparison_seeds,
                )

            for name, seed_embds in (("Sentence A", seed_a), ("Sentence B", seed_b)):
                if seed_embds is not None and len(seed_embds) > 1:
                    st.caption(
                        f"APE split the seed for {name} into {len(seed_embds)} propositions: "
                        "each one highlights its best match. Seeds work best when they are "
                        "a single atomic proposition."
                    )