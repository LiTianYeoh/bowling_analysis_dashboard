import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

NUM_SIMULATIONS = 10000

# ==========================================
# CSS
# ==========================================
CSS_MARKDOWN = """
<style>
/* Green primary button */
div.stButton:has(button[kind="primary"]) button {
    background-color: #28a745;
    color: white;
    border: 1px solid #28a745;
}

div.stButton:has(button[kind="primary"]) button:hover {
    background-color: #218838;
    border-color: #1e7e34;
    color: white;
}

/* Red secondary button */
div.stButton:has(button[kind="secondary"]) button {
    background-color: #dc3545;
    color: white;
    border: 1px solid #dc3545;
}

div.stButton:has(button[kind="secondary"]) button:hover {
    background-color: #c82333;
    border-color: #bd2130;
    color: white;
}
</style>
"""

# ==========================================
# REQUIRED DATA FORMAT
# ==========================================

REQUIRED_COLUMNS_DTYPES = {
    "round_id": str,
    "date": str,
    "name": str,
    "playstyle": str,
    "f1b1": int,
    "f1b2": int,
    "f2b1": int,
    "f2b2": int,
    "f3b1": int,
    "f3b2": int,
    "f4b1": int,
    "f4b2": int,
    "f5b1": int,
    "f5b2": int,
    "f6b1": int,
    "f6b2": int,
    "f7b1": int,
    "f7b2": int,
    "f8b1": int,
    "f8b2": int,
    "f9b1": int,
    "f9b2": int,
    "f10b1": int,
    "f10b2": int,
    "f10b3": int,
    "final_score": int
}

REQUIRED_COLUMNS = REQUIRED_COLUMNS_DTYPES.keys()


# ==========================================
# BOWLING SCORE CALCULATION
# ==========================================


def calculate_score(row):
    """
    Calculate a bowling game's final score, number of strikes,
    and number of spares.

    Returns
    -------
    tuple
        (calculated_score, num_strikes, num_spares)
    """

    total_score = 0
    num_strikes = 0
    num_spares = 0

    bonus_score_status = [0, 0]

    # Frames 1 to 9
    for frame in range(1, 10):
        ball1 = row[f"f{frame}b1"]
        ball2 = row[f"f{frame}b2"]

        # Count strikes and spares
        if ball1 == 10:
            num_strikes += 1
        elif ball1 + ball2 == 10:
            num_spares += 1

        # Calculate score, including bonuses
        total_score += (
            ball1 * (1 + bonus_score_status[0])
            + ball2 * (1 + bonus_score_status[1])
        )

        # Update bonuses for subsequent balls
        if ball1 == 10:
            # Strike
            bonus_score_status = [
                1 + bonus_score_status[1],
                1,
            ]

        elif ball1 + ball2 == 10:
            # Spare
            bonus_score_status = [1, 0]

        else:
            bonus_score_status = [0, 0]

    # Frame 10
    f10b1 = row["f10b1"]
    f10b2 = row["f10b2"]
    f10b3 = row["f10b3"]

    # Count strikes and spares in frame 10.
    # A strike is a ball knocking down all 10 pins.
    if f10b1 == 10:
        num_strikes += 1
    elif f10b1 + f10b2 == 10:
        num_spares += 1

    # # The second ball can also be a strike if the first
    # # ball was a strike.
    # if f10b1 == 10 and f10b2 == 10:
    #     num_strikes += 1
    # elif f10b1 == 10 and f10b2 + f10b3 == 10:
    #     num_spares += 1

    frame10_score = f10b1 + f10b2 + f10b3

    total_score += (
        f10b1 * bonus_score_status[0]
        + f10b2 * bonus_score_status[1]
        + frame10_score
    )

    return total_score, num_strikes, num_spares

# ==========================================
# DATA VALIDATION
# ==========================================

def validate_columns(df):
    """
    Check whether the uploaded DataFrame has the required columns.

    Returns
    -------
    tuple
        (is_valid, missing_columns, unexpected_columns)
    """

    # Normalise column headers
    df.columns = df.columns.astype(str).str.strip()

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    unexpected_columns = [
        col for col in df.columns
        if col not in REQUIRED_COLUMNS
    ]

    is_valid = len(missing_columns) == 0 and not df.empty

    return is_valid, missing_columns, unexpected_columns


def check_score_consistency(df):
    """
    Compare each row's calculated score against final_score.

    Returns
    -------
    pandas.DataFrame
        Rows with missing/invalid scores or mismatched final scores.
    """

    df = df.copy()

    numeric_columns = [
        col for col in REQUIRED_COLUMNS
        if col.startswith("f") and col != "final_score"
    ]

    numeric_columns.append("final_score")

    # Convert score columns to numeric values.
    # Invalid values become NaN and are reported below.
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_mask = df[numeric_columns].isna().any(axis=1)

    # Also reject non-integer or negative ball scores,
    # and ball scores above 10.
    ball_columns = [
        col for col in numeric_columns
        if col != "final_score"
    ]

    invalid_ball_mask = (
        (df[ball_columns] < 0)
        | (df[ball_columns] > 10)
        | (df[ball_columns] % 1 != 0)
    ).any(axis=1)

    invalid_score_mask = (
        (df["final_score"] < 0)
        | (df["final_score"] > 300)
        | (df["final_score"] % 1 != 0)
    )

    invalid_mask = (
        invalid_mask
        | invalid_ball_mask
        | invalid_score_mask
    )

    # Calculate scores only for rows with usable numeric data.
    valid_rows = ~invalid_mask

    
    # Calculate score and strike/spare counts
    df["calculated_score"] = pd.NA
    df["num_strikes"] = pd.NA
    df["num_spares"] = pd.NA

    valid_rows = ~invalid_mask

    if valid_rows.any():
        results = df.loc[valid_rows].apply(
            calculate_score,
            axis=1,
            result_type="expand",
        )

        results.columns = [
            "calculated_score",
            "num_strikes",
            "num_spares",
        ]

        df.loc[valid_rows, results.columns] = results

    # Identify mismatches
    mismatch_mask = (
        valid_rows
        & (
            df["final_score"]
            != pd.to_numeric(
                df["calculated_score"],
                errors="coerce",
            )
        )
    )

    error_mask = invalid_mask | mismatch_mask

    error_df = df.loc[error_mask].copy()

    error_df["issue"] = "Score mismatch"
    error_df.loc[invalid_mask, "issue"] = (
        "Missing or invalid score values"
    )

    display_columns = [
        "round_id",
        "date",
        "name",
        "playstyle",
        "final_score",
        "calculated_score",
        "num_strikes",
        "num_spares",
        "issue",
    ]

    return error_df[display_columns]


# ==========================================
# DATA ANALYSIS
# ==========================================

def cal_plph_matrix(frame_ball_dict):
    plph_matrix = np.zeros([10,11])

    for frame in range(1, 10):
        ball1 = frame_ball_dict[f"f{frame}b1"]
        ball2 = frame_ball_dict[f"f{frame}b2"]

        plph_matrix[0, ball1] += 1

        if ball1 != 10: #not strike, ball2 is valid
            plph_matrix[ball1, ball2] += 1

    # handle frame 10
    f10b1 = frame_ball_dict[f"f10b1"]
    f10b2 = frame_ball_dict[f"f10b2"]
    f10b3 = frame_ball_dict[f"f10b3"]

    plph_matrix[0, f10b1] += 1

    if f10b1 != 10: #not strike, ball2 is valid
        plph_matrix[f10b1, f10b2] += 1

        if f10b1 + f10b2 == 10: #spare
            plph_matrix[0, f10b3] += 1

    else:   #strike
        plph_matrix[0, f10b2] += 1

        if f10b2 == 10: #double strike
            plph_matrix[0, f10b3] += 1
        else:
            plph_matrix[f10b2, f10b3] += 1

    return plph_matrix


def plot_plph_pmf_matrix(plph_pmf_matrix):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("PMF funtion at various #Pin_Left", fontsize=16, fontweight="bold")

    im = ax.imshow(plph_pmf_matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Probability")

    # Display values inside each cell
    for i in range(plph_pmf_matrix.shape[0]):
        for j in range(11-i):
            ax.text(
                j, i, f"{plph_pmf_matrix[i, j]:.2f}",
                ha="center", va="center", color="black"
            )

    # ax.set_title(title_str)

    ax.set_xlabel("#Pin_Hit")
    ax.set_xticks(np.arange(0, 11, 1))

    ax.set_ylabel("#Pin_Left")
    ax.set_yticks(
        ticks = np.arange(plph_pmf_matrix.shape[0]),
        labels = [f"{10-i}" for i in range(0, plph_pmf_matrix.shape[0])]
    )

    return fig, ax


def simulate_pin_hit(plph_pmf_matrix, pin_left):
    if pin_left == 0:
        return 0

    curr_pl_pmf = plph_pmf_matrix[10-pin_left, 0:pin_left+1]

    if np.sum(curr_pl_pmf) < 0.001:
        print(f"PMF at #Pin_Left={pin_left} < 0.001 (no_data), default to pin_hit = 0")
        return 0

    elif np.sum(curr_pl_pmf) < 0.999:
        print(f"!!! Invalid PMF_sum = {np.sum(curr_pl_pmf)} at #Pin_Left={pin_left}, default to pin_hit = 0")
        return 0

    else:
        #valid pmf, proceed to simulate
        return np.random.choice(len(curr_pl_pmf), p=curr_pl_pmf)


def simulate_round_score(plph_pmf_matrix):
    sim_frame_ball_dict = {}

    # simulate frame 1 to 9
    for curr_frame in range(1, 10):
        curr_pin_left = 10

        for curr_ball in range(1,3):

            sim_pin_hit = simulate_pin_hit(plph_pmf_matrix, curr_pin_left)
            sim_frame_ball_dict[f"f{curr_frame}b{curr_ball}"] = sim_pin_hit

            curr_pin_left = curr_pin_left - sim_pin_hit
            

    # simulate frame 10
    # f10b1
    sim_f10b1 = simulate_pin_hit(plph_pmf_matrix, 10)
    sim_frame_ball_dict[f"f10b1"] = sim_f10b1

    if sim_f10b1 == 10:  #strike
        sim_f10b2 = simulate_pin_hit(plph_pmf_matrix, 10)
        sim_frame_ball_dict[f"f10b2"] = sim_f10b2

        if sim_f10b2 == 10: #double strike
            sim_f10b3 = simulate_pin_hit(plph_pmf_matrix, 10)
        else:
            sim_f10b3 = simulate_pin_hit(plph_pmf_matrix, 10-sim_f10b2)

        sim_frame_ball_dict[f"f10b3"] = sim_f10b3

    else:
        sim_f10b2 = simulate_pin_hit(plph_pmf_matrix, 10-sim_f10b1)
        sim_frame_ball_dict[f"f10b2"] = sim_f10b2

        if sim_f10b1 + sim_f10b2 == 10: #spare
            sim_f10b3 = simulate_pin_hit(plph_pmf_matrix, 10)
        else:
            sim_f10b3 = 0

        sim_frame_ball_dict[f"f10b3"] = sim_f10b3
    
    simulated_final_score, simulated_num_strike, simulated_num_spare = calculate_score(sim_frame_ball_dict)

    # print(sim_frame_ball_dict)
    # print(f"Simulated score: {simulated_final_score}")

    return simulated_final_score, simulated_num_strike, simulated_num_spare


def fig_to_buf(fig):
    """Convert a Matplotlib figure into PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def calculate_monte_carlo_stats(final_score_list):
    result = {"Avg Score": np.round(np.mean(final_score_list), 2)}

    for q in [0.05, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.9, 0.95]:
        result[f"q{q:.2f}"] = np.quantile(final_score_list, q)

    return result




def do_analysis(target_df):

    # calculate plph_pmf_matrix
    plph_matrix_all = np.zeros([10,11])

    for _, round_row in target_df.iterrows():
        plph_matrix_all += cal_plph_matrix(round_row)

    row_sums = plph_matrix_all.sum(axis=1, keepdims=True)
    plph_pmf_matrix = plph_matrix_all / row_sums
    plph_pmf_matrix = np.nan_to_num(plph_pmf_matrix, nan=0)

    # plot heatmap
    plph_pmf_matrix_actual_fig, ax = plot_plph_pmf_matrix(plph_pmf_matrix)

    ax.set_title(f"Number of Rounds = {target_df.shape[0]}")
    ax.set_ylabel("#Pin_Left (#samples)")
    ax.set_yticks(
        ticks = np.arange(plph_pmf_matrix.shape[0]),
        labels = [f"{10-i} ({int(row_sums[i, 0])})" for i in range(0, row_sums.shape[0])]
    )

    heatmap_buf = fig_to_buf(plph_pmf_matrix_actual_fig)



    # monte carlo performance
    simulated_final_scores = []
    simulated_total_num_strike = 0
    simulated_total_num_spare = 0

    for _ in range(NUM_SIMULATIONS):
        simulated_final_score, simulated_num_strike, simulated_num_spare = simulate_round_score(plph_pmf_matrix)
        
        simulated_final_scores.append(simulated_final_score)
        simulated_total_num_strike += simulated_num_strike
        simulated_total_num_spare += simulated_num_spare

    simulated_avg_num_strike = round(simulated_total_num_strike/NUM_SIMULATIONS, 2)
    simulated_avg_num_spare = round(simulated_total_num_spare/NUM_SIMULATIONS, 2)

    monte_carlo_result = {
        "simulated_final_scores": simulated_final_scores,
        "simulated_avg_num_strike": simulated_avg_num_strike,
        "simulated_avg_num_spare": simulated_avg_num_spare
    }

    return heatmap_buf, monte_carlo_result