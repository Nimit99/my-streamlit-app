import streamlit as st
import torch
from transformers import pipeline
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import re

# =========================================
# 1) Slack Text (Raw) - Example
# =========================================
slack_text = """
Jharad 2025-01-01 09:27
I love testing things out! I love college football and college basketball, but in general, I love college sports!

KEMI Levi 2025-01-01 16:18
College test?

Pangia APP 2025-01-01 17:03
:rotating_light:Event: Some Slack Bot Prompt

Jharad 2025-01-02 06:04
renamed the channel from “welcome” to “welcome-edit”

Pangia APP 2025-01-02 06:12
:rotating_light:Event: Another Slack Bot Prompt

Nassir Mohamud 2025-01-04 08:05
renamed the channel from “welcome-edit” to “testing-channel”

Nimit 2025-01-06 09:49
joined #testing-channel.

Hello everyone. This is a test message

Nimit 2025-01-06 09:50
Are you guys free for a call later?
"""

# =========================================
# 2) Helper Functions
# =========================================
def parse_slack_lines(slack_text):
    """
    Splits text into lines.
    """
    return slack_text.strip().split('\n')

def extract_user_and_datetime(line):
    """
    Naive attempt to parse "Username YYYY-MM-DD HH:MM" from each line.
    Returns (user_name, datetime_object) or (None, None) if it fails.
    """
    pattern = r"^(.*)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})$"
    match = re.match(pattern, line.strip())
    if match:
        user_part = match.group(1).strip()
        datetime_part = match.group(2).strip()
        try:
            dt_obj = datetime.strptime(datetime_part, "%Y-%m-%d %H:%M")
            return user_part, dt_obj
        except ValueError:
            pass
    return None, None

def get_user_messages(lines):
    """
    Returns a list of dicts, each with:
       {
         "user": <username>,
         "datetime": <datetime_obj>,
         "text": <the actual message line after the user/time line>
       }
    We assume each user/time line is followed by 0 or more lines of text that belong to that user,
    until we reach the next user/time line.
    """
    user_messages = []
    current_user = None
    current_dt = None
    buffer_text = []

    for line in lines:
        # Check if this line has "Username YYYY-MM-DD HH:MM"
        user, dt_obj = extract_user_and_datetime(line)
        if user and dt_obj:
            # If we had a previous user, store their buffered text
            if current_user is not None and buffer_text:
                user_messages.append({
                    "user": current_user,
                    "datetime": current_dt,
                    "text": "\n".join(buffer_text)
                })
                buffer_text = []

            current_user = user
            current_dt = dt_obj
        else:
            buffer_text.append(line)

    # End of lines, if something is left in buffer
    if current_user is not None and buffer_text:
        user_messages.append({
            "user": current_user,
            "datetime": current_dt,
            "text": "\n".join(buffer_text)
        })
    
    return user_messages

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

def classify_message_types(text):
    """
    Returns how many lines are questions vs. statements.
    For demonstration, we do a simple check:
       - If a line ends in '?', it's a question.
       - Otherwise, it's a statement.
    """
    lines = text.split('\n')
    question_count = sum([1 for l in lines if l.strip().endswith('?')])
    statement_count = len(lines) - question_count
    return question_count, statement_count

def compute_sentiment(text, sentiment_pipeline):
    """
    We average the sentiment scores of each line.
    If line label=POSITIVE => +score, if NEGATIVE => -score.
    """
    lines = text.split('\n')
    if not lines:
        return 0.0
    
    scores = []
    for l in lines:
        result = sentiment_pipeline(l[:512])[0]
        if result['label'] == 'POSITIVE':
            scores.append(result['score'])
        else:
            scores.append(-result['score'])
    if not scores:
        return 0.0
    return np.mean(scores)

def days_since(date_obj, reference_date):
    """Return how many days between reference_date and date_obj."""
    delta = reference_date - date_obj
    return delta.days

# =========================================
# 3) Build Feature Data
# =========================================
def build_feature_data(user_messages, sentiment_pipeline):
    """
    For each user, we compute:
       - total_messages
       - days_since_last_message
       - total_questions
       - total_statements
       - average_sentiment
       - drop_off (mocked)
    """
    from collections import defaultdict
    user_data = defaultdict(list)
    for msg_dict in user_messages:
        user_data[msg_dict['user']].append(msg_dict)
    
    # The reference date is the last message's date (treated as 'today')
    reference_date = max([m['datetime'] for m in user_messages]) if user_messages else datetime.now()

    rows = []
    for user, msgs in user_data.items():
        # Sort by date
        msgs_sorted = sorted(msgs, key=lambda x: x['datetime'])
        
        total_msgs = len(msgs_sorted)
        last_msg_date = msgs_sorted[-1]['datetime']
        gap_days = days_since(last_msg_date, reference_date)
        
        total_questions = 0
        total_statements = 0
        sentiment_list = []
        
        for m in msgs_sorted:
            q_count, s_count = classify_message_types(m['text'])
            total_questions += q_count
            total_statements += s_count
            sentiment_list.append(compute_sentiment(m['text'], sentiment_pipeline))
        
        avg_sentiment = np.mean(sentiment_list) if sentiment_list else 0.0

        # Mock label: if user hasn't posted in >2 days => drop_off=1, else 0
        drop_off_label = 1 if gap_days > 2 else 0

        rows.append({
            "user": user,
            "total_messages": total_msgs,
            "days_since_last_message": gap_days,
            "total_questions": total_questions,
            "total_statements": total_statements,
            "average_sentiment": avg_sentiment,
            "drop_off": drop_off_label
        })

    return pd.DataFrame(rows)

# =========================================
# 4) Streamlit App
# =========================================
st.title("Slack Drop-Off Analysis (No /going or /maybe counts)")

# -- Parse lines
lines = parse_slack_lines(slack_text)

# -- Extract user messages
all_user_messages = get_user_messages(lines)

# -- Load sentiment pipeline
sentiment_pipe = load_sentiment_pipeline()

# -- Build feature data
df = build_feature_data(all_user_messages, sentiment_pipe)

st.subheader("Feature Data (Mock Labels)")
st.write(df)

# Train a simple logistic regression
X = df[["total_messages", 
        "days_since_last_message",
        "total_questions",
        "total_statements",
        "average_sentiment"]]
y = df["drop_off"]

model = LogisticRegression()
model.fit(X, y)

st.write("Trained Model Coefficients:")
coeff_df = pd.DataFrame({
    "feature": X.columns,
    "coef": model.coef_[0]
})
st.write(coeff_df)
st.write(f"Intercept: {model.intercept_[0]}")

# Let user pick a user to see drop-off prediction
user_options = df["user"].unique().tolist()
selected_user = st.selectbox("Select a user to predict drop-off:", user_options)

if selected_user:
    # Filter the row for that user
    user_row = df[df["user"] == selected_user].iloc[0]
    
    # Prepare the row for prediction
    X_test = np.array([[
        user_row["total_messages"],
        user_row["days_since_last_message"],
        user_row["total_questions"],
        user_row["total_statements"],
        user_row["average_sentiment"]
    ]])
    
    dropoff_prob = model.predict_proba(X_test)[0, 1]  # Probability user=1 (drop off)
    st.subheader(f"Prediction for {selected_user}")
    st.write(f"**Total Messages**: {user_row['total_messages']}")
    st.write(f"**Days Since Last Message**: {user_row['days_since_last_message']}")
    st.write(f"**Total Questions**: {user_row['total_questions']}")
    st.write(f"**Total Statements**: {user_row['total_statements']}")
    st.write(f"**Average Sentiment**: {user_row['average_sentiment']:.2f}")
    st.write(f"**Drop-Off Probability**: {dropoff_prob*100:.2f}%")

    # -----------------------------------------
    # 5) Advice Section
    # -----------------------------------------
    st.markdown("### Advice to Reduce Drop-Off Risk")
    advice_points = []

    # Simple logic to generate 2–3 suggestions:

    # 1) Sentiment advice
    if user_row["average_sentiment"] < 0:
        advice_points.append(
            "Your messages show a predominantly **negative sentiment**. "
            "Consider reaching out for support or talking to an advisor. "
            "Positive interaction can boost your engagement."
        )

    # 2) Encouraging more frequent participation
    if user_row["days_since_last_message"] > 1:
        advice_points.append(
            "You haven't posted in the channel for a while. "
            "Try to participate more frequently—ask questions or share insights. "
            "Consistency can help you stay connected."
        )
    
    # 3) Asking more questions vs. statements
    if user_row["total_questions"] == 0:
        advice_points.append(
            "It looks like you're not asking any questions. "
            "Asking questions is a great way to learn, engage with peers, "
            "and stay active in the conversation!"
        )

    # If we didn't get any advice from the above, we can add a general one:
    if not advice_points:
        advice_points.append(
            "Great job staying active! Keep an eye on your sentiment and engagement over time."
        )

    # Display bullet points
    for idx, tip in enumerate(advice_points, 1):
        st.markdown(f"**{idx}. {tip}**")

st.write("---")
st.write("*Note: This is a demo with mocked 'drop_off' labels. In production, you would use real churn data.*")
