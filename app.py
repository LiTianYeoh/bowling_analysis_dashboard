import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from utils import (
    CSS_MARKDOWN,
    REQUIRED_COLUMNS_DTYPES, REQUIRED_COLUMNS,
    validate_columns,
    check_score_consistency,
    do_analysis,
    NUM_SIMULATIONS,
    calculate_monte_carlo_stats
)


# ==========================================
# APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Bowling App",
    page_icon="📊",
    layout="wide"
)

# st.title("Bowling Performance Dashboard")
# st.write(
#     "Upload your bowling dataset and perform data analysis "
#     "using the sections in the sidebar."
# )

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Bowling Performance Dashboard")

page = st.sidebar.radio(
    "Section",
    [
        "Upload Data",
        "Create Filter Conditions",
        "Performance Analysis"
    ],
)

# ==========================================
# CSS
# ==========================================

st.markdown(CSS_MARKDOWN,unsafe_allow_html=True,)

# ==========================================
# Initialise session_state
# ==========================================

if "filter_conditions" not in st.session_state:
    st.session_state["filter_conditions"] = []

if "filter_results" not in st.session_state:
    st.session_state["filter_results"] = {}

if "pending_filter" not in st.session_state:
    st.session_state["pending_filter"] = None

# ==========================================
# PART 1: Upload Data
# ==========================================

if page == "Upload Data":

    st.header("Upload Data")

    st.write(
        "Upload your bowling dataset and check "
        "whether the recorded final scores are correct."
    )

    with st.expander("View Required Data Format"):

        st.write("Your file must contain these columns:")

        st.code(", ".join(REQUIRED_COLUMNS))

        st.caption(
            f"Total required columns: {len(REQUIRED_COLUMNS)}"
        )

        st.divider()

        st.subheader("Data Entry Instructions")

        st.markdown("""
        - **Strike:** Record `b1 = 10` and `b2 = 0`.
        - **Spare:** Record the first and second rolls such that `b1 + b2 = 10` (e.g., `b1 = 9`, `b2 = 1`).
        """)

        # st.info(
        #     "The score consistency check compares the calculated "
        #     "score against the supplied final_score."
        # )

        with open("sample_data.csv", "rb") as f:
            sample_file = f.read()

        st.download_button(
            label="Download Sample CSV",
            data=sample_file,
            file_name="sample_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=["csv", "xlsx"],
        key="bowling_upload",
    )

    st.warning(
        "Uploading a new data file will reset all existing filter conditions."
    )

    if uploaded_file is not None:

        try:
            # Read the uploaded file
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(
                    uploaded_file,
                    header=0,
                    dtype=REQUIRED_COLUMNS_DTYPES,
                )
            else:
                df = pd.read_excel(
                    uploaded_file,
                    header=0,
                    dtype=REQUIRED_COLUMNS_DTYPES,
                )

            # Validate the structure
            is_valid, missing_columns, unexpected_columns = (
                validate_columns(df)
            )

            if missing_columns:

                st.error(
                    "Upload failed: required columns are missing."
                )

                st.write("**Missing columns:**")
                st.write(", ".join(missing_columns))

            elif df.empty:

                st.error(
                    "The uploaded file contains no data rows."
                )

            else:

                # Keep only the required columns, in the
                # expected order.
                df = df[REQUIRED_COLUMNS].copy()

                st.success(
                    "The file structure is valid."
                )

                if unexpected_columns:
                    st.warning(
                        "Extra columns were excluded: "
                        + ", ".join(unexpected_columns)
                    )

                # ------------------------------------------
                # DATA PREVIEW
                # ------------------------------------------

                st.subheader("Data Preview")

                st.dataframe(
                    df.head(10),
                    hide_index=True,
                    use_container_width=True
                )

                # ------------------------------------------
                # DATASET SUMMARY
                # ------------------------------------------

                st.subheader("Dataset Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Rows", len(df))
                col2.metric("Total Columns", len(df.columns))

                col3.metric(
                    "Missing Values",
                    int(df.isna().sum().sum()),
                )

                # ------------------------------------------
                # SCORE CONSISTENCY CHECK
                # ------------------------------------------

                st.subheader("Score Consistency Check")

                with st.spinner(
                    "Checking calculated scores..."
                ):
                    error_df = check_score_consistency(df)

                if error_df.empty:

                    st.success(
                        "All rows passed the score consistency "
                        "check!"
                    )

                    # Save the validated data
                    st.session_state["uploaded_df"] = df
                    st.session_state["uploaded_filename"] = uploaded_file.name
                    st.session_state["upload_valid"] = True

                    st.session_state["filter_conditions"] = []
                    st.session_state["filter_results"] = {}

                else:

                    mismatch_count = len(error_df)

                    st.error(
                        f"Found {mismatch_count} row(s) with "
                        "invalid score values or score mismatches."
                    )

                    st.dataframe(
                        error_df,
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.warning(
                        "Please correct the data and upload the "
                        "updated file before proceeding to analysis."
                    )

                    # Do not allow a dataset that fails the
                    # sanity check to be used in Parts 2 and 3.
                    st.session_state.pop("uploaded_df", None)
                    st.session_state["upload_valid"] = False

        except Exception as e:

            st.error(
                f"Unable to read or validate the file: {e}"
            )

# ==========================================
# PART 2: FILTER SELECTOR
# ==========================================
elif page == "Create Filter Conditions":

    st.header("Create Filter Conditions")

    if (
        "uploaded_df" not in st.session_state
        or not st.session_state.get("upload_valid", False)
    ):
        st.info("Please upload a dataset first.")

    else:
        df = st.session_state["uploaded_df"]
        filename = st.session_state["uploaded_filename"]

        st.write("**Raw data file:**")
        st.info(f"📄 {filename}")

        # ==========================================
        # PART 2A: FILTER SELECTOR
        # ==========================================

        st.subheader("Manage Filter Conditions (Supports Multiple Filter Conditions)")

        # Convert date column to datetime for date filtering
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Get available options from the uploaded data
        available_names = sorted(df["name"].dropna().unique().tolist())
        available_playstyles = sorted(df["playstyle"].dropna().unique().tolist())

        min_date = df["date"].min().date()
        max_date = df["date"].max().date()

        # filter selector        
        start_date_col, end_date_col, player_name_col, playstyle_col = st.columns(4)

        with start_date_col:
            start_date = st.date_input(
                "Start date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="new_filter_start_date",
            )

        with end_date_col:
            end_date = st.date_input(
                "End date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="new_filter_end_date",
            )

        with player_name_col:
            player_name_selection = st.multiselect(
                "Player Name",
                options=available_names,
                default=[],
                key="new_filter_player_names",
            )

        with playstyle_col:
            playstyle_selection = st.multiselect(
                "Playstyle",
                options=available_playstyles,
                default=[],
                key="new_filter_playstyles",
            )

        # add filter button
        filter_name_col, button_col = st.columns(2)

        with filter_name_col:
            filter_name = st.text_input(
                "Filter condition name",
                placeholder="e.g. Player A - Normal Play",
                key="new_filter_name",
            )

        with button_col:
            st.write("")  # Small vertical spacing
            st.write("")
            create_filter = st.button(
                "Create New Filter Condition",
                type="primary",
                use_container_width=True,
            )

        # Save the filter condition
        if create_filter:
            if start_date > end_date:
                st.warning("Start date must not be later than end date.")

            elif len(player_name_selection) == 0:
                st.warning("Please select at least 1 Player Name")

            elif len(playstyle_selection) == 0:
                st.warning("Please select at least 1 Playstyle")

            elif not filter_name.strip():
                st.warning("Please enter a name for this filter condition.")

            elif any(
                f["filter_name"] == filter_name.strip()
                for f in st.session_state["filter_conditions"]
            ):
                st.warning(f"A filter condition with name '{filter_name}' already exists.")

            else:
                new_filter = {
                    "filter_name": filter_name.strip(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "player_names": player_name_selection,
                    "playstyles": playstyle_selection,
                }

                filter_mask = (
                    pd.to_datetime(df["date"], errors="coerce").dt.date.between(
                        new_filter["start_date"],
                        new_filter["end_date"],
                    )
                    & df["name"].isin(new_filter["player_names"])
                    & df["playstyle"].isin(new_filter["playstyles"])
                )
                target_df = df[filter_mask]
                num_rounds = target_df.shape[0]

                if num_rounds == 0:
                    st.warning(f"Current filter condition has no data.")

                else:
                    st.session_state["pending_filter"] = {
                        "filter_conds": new_filter,
                        "num_rounds": num_rounds,
                        "target_df": target_df
                    }

        # New filter confirmation
        if st.session_state["pending_filter"] is not None:
            with st.expander("Confirm Filter Condition", expanded=True):
                pending_filter = st.session_state["pending_filter"]

                st.write("Please review your filter condition:")

                st.write(f"**Filter Name:** {pending_filter['filter_conds']['filter_name']}")
                st.write(f"**Start Date:** {pending_filter['filter_conds']['start_date'] or 'Any'}")
                st.write(f"**End Date:** {pending_filter['filter_conds']['end_date'] or 'Any'}")
                st.write(f"**Player Names:** {', '.join(pending_filter['filter_conds']['player_names']) or 'All'}")
                st.write(f"**Playstyles:** {', '.join(pending_filter['filter_conds']['playstyles']) or 'All'}")

                st.metric("Number of Rounds: ", pending_filter['num_rounds'])

                st.warning("It takes around 30s to run the analysis (monte carlo simulation)")

                col1, col2 = st.columns(2)

                with col1:
                    confirm_pending_filter = st.button("Confirm", type="primary", use_container_width=True)
                    
                with col2:
                    cancel_pending_filter = st.button("Cancel", type="secondary", use_container_width=True)

                if confirm_pending_filter:
                    st.write("Performing analysis, please wait...")

                    # do analysis on the filter condition
                    heatmap_buf, monte_carlo_result = do_analysis(pending_filter["target_df"])

                    # Save the filter condition and filter results
                    st.session_state["filter_conditions"].append(pending_filter["filter_conds"])
                    st.session_state["filter_results"][pending_filter["filter_conds"]["filter_name"]] = {
                        "num_rounds": pending_filter['num_rounds'],
                        "heatmap_buf": heatmap_buf,
                        "monte_carlo_result": monte_carlo_result
                    }

                    st.session_state["pending_filter"] = None
                    st.rerun()

                if cancel_pending_filter:
                    st.session_state["pending_filter"] = None
                    st.rerun()


        # Display all filter conditions
        if st.session_state["filter_conditions"]:
            st.subheader("All Filter Conditions")

            filter_table = [
                {
                    "Filter Name": condition["filter_name"],
                    "Start Date": condition["start_date"],
                    "End Date": condition["end_date"],
                    "Player Names": ", ".join(condition["player_names"]),
                    "Playstyles": ", ".join(condition["playstyles"]),
                    "#Rounds": st.session_state["filter_results"][condition["filter_name"]]["num_rounds"],
                } for condition in st.session_state["filter_conditions"]
            ]

            filter_df = pd.DataFrame(filter_table)

            st.dataframe(
                filter_df,
                hide_index=True,
                use_container_width=True,
            )

            # button to remove filter condition
            filter_name_remove_col, button_remove_col = st.columns(2)

            with filter_name_remove_col:
                all_filter_names = [conds["filter_name"] for conds in st.session_state["filter_conditions"]]
                filter_name_remove = st.selectbox(
                    "Filter Name",
                    options=all_filter_names,
                    key="remove_filter_player_names",
                )

            with button_remove_col:
                st.write("")  # Small vertical spacing
                st.write("")
                remove_filter = st.button(
                    "Remove Filter Condition",
                    type="secondary",
                    use_container_width=True,
                )

            # Remove the filter condition
            if remove_filter:
                st.session_state["filter_conditions"] = [
                    f for f in st.session_state["filter_conditions"]
                    if f["filter_name"] != filter_name_remove
                ]
                st.session_state["filter_results"].pop(filter_name_remove, None)
                st.rerun()



# ==========================================
# PART 3: FILTER SELECTOR
# ==========================================
elif page == "Performance Analysis":

    st.header("Performance Analysis")

    if (
        "uploaded_df" not in st.session_state
        or not st.session_state.get("upload_valid", False)
    ):
        st.info("Please upload a dataset first.")

    elif len(st.session_state["filter_conditions"]) == 0:
        st.info("Please create at least 1 filter condition first.")

    else:

        # display plph_pmf_matrix heatmap
        st.subheader("Pin_Left vs Pin_Hit PMF Matrix Heatmap")

        all_filter_names = [conds["filter_name"] for conds in st.session_state["filter_conditions"]]

        left_filter_col, right_filter_col = st.columns(2)

        with left_filter_col:
            left_filter_cond = st.selectbox(
                "Filter Name 1",
                options=all_filter_names,
                key="left_filter_cond_name",
            )

            if left_filter_cond:
                left_heatmap_buf = st.session_state["filter_results"][left_filter_cond]["heatmap_buf"]
                st.image(left_heatmap_buf, use_column_width=True)

        with right_filter_col:
            right_filter_cond = st.selectbox(
                "Filter Name 2",
                options=all_filter_names,
                key="right_filter_cond_name",
            )

            if right_filter_cond:
                right_heatmap_buf = st.session_state["filter_results"][right_filter_cond]["heatmap_buf"]
                st.image(right_heatmap_buf, use_column_width=True)


        # display monte_carlo_result
        st.subheader(f"Monte Carlo Result ({NUM_SIMULATIONS} simulations)")

        # Aggregate Monte Carlo result
        score_sim_dict = {}
        score_metric_list = []
        
        for filter_name, result_dict in st.session_state["filter_results"].items():
            monte_carlo_result = result_dict["monte_carlo_result"]
            score_sim_dict[filter_name] = monte_carlo_result["simulated_final_scores"]

            score_metric_dict = {
                "Filter Name": filter_name,
                "Avg #Strike": monte_carlo_result["simulated_avg_num_strike"],
                "Avg #Spare": monte_carlo_result["simulated_avg_num_spare"]
            }
            score_metric_dict.update(calculate_monte_carlo_stats(monte_carlo_result["simulated_final_scores"]))

            score_metric_list.append(score_metric_dict)

        score_metric_df = pd.DataFrame(score_metric_list)

        st.dataframe(
            score_metric_df,
            hide_index=True,
            use_container_width=True,
        )

        # kde plot
        kde_fig = go.Figure()

        # Common x-axis range for all KDE curves
        all_values = np.concatenate([np.asarray(values) for values in score_sim_dict.values()])
        x = np.linspace(all_values.min(), all_values.max(), 300)

        for name, values in score_sim_dict.items():
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]

            if len(values) < 2 or np.std(values) == 0:
                continue

            kde = gaussian_kde(values, bw_method="scott")

            # Increase bandwidth to smooth the curve
            kde.set_bandwidth(bw_method=kde.factor * 3.0)

            kde_fig.add_trace(
                go.Scatter(
                    x=x,
                    y=kde(x),
                    mode="lines",
                    name=str(name),
                    line=dict(width=2)
                )
            )

        kde_fig.update_xaxes(range=[0, 300])

        kde_fig.update_layout(
            title="Round Score Distribution Comparison",
            xaxis_title="Value",
            yaxis_title="Density",
            template="plotly_white"
        )

        st.plotly_chart(kde_fig, use_container_width=True)


