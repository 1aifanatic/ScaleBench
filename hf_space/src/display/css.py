"""CSS styling for leaderboard."""

custom_css = """
.markdown-text {
    font-size: 16px !important;
}

#leaderboard-table {
    margin-top: 15px;
}

/* Limit model column width */
#leaderboard-table td:nth-child(1),
#leaderboard-table th:nth-child(1) {
    max-width: 300px;
    overflow: auto;
    white-space: nowrap;
}

.tab-buttons button {
    font-size: 18px;
}

#citation-button span {
    font-size: 14px !important;
}

#citation-button textarea {
    font-size: 14px !important;
}

/* Mode highlighting */
.baseline-row {
    background-color: rgba(100, 149, 237, 0.1);
}

.scaledown-row {
    background-color: rgba(50, 205, 50, 0.1);
}

/* Positive delta (improvement) */
.positive-delta {
    color: green;
    font-weight: bold;
}

/* Negative delta (regression) */
.negative-delta {
    color: red;
    font-weight: bold;
}
"""
