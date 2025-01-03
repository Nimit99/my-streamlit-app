import streamlit as st
import torch
from transformers import pipeline

# =========================================
# 1) Slack Text (Raw)
# =========================================
slack_text = """
Jharad 9:27 AM
I love testing things out! I love college football and college basketball, but in general, I love college sports!

KEMI Levi 4:18 PM
College test

Pangia APP 5:03 PM
:rotating_light:Event: Sports and Founders in SF
Please reply with one of the following commands: /going, /not-going, or /maybe

Pangia APP 8:13 AM
:rotating_light:Event: Basketball tournament...
Please reply with one of the following commands: /going, /not-going, or /maybe

Jharad 6:04 PM
renamed the channel from “welcome” to “welcome-edit”

Pangia APP 6:12 PM
:rotating_light:Event: Basketball Event...
Please reply with one of the following commands: /going, /not-going, or /maybe

Nassir Mohamud 8:05 AM
renamed the channel from “welcome-edit” to “testing-channel”

Nimit 9:49 AM
joined #testing-channel.

/not-going
/going
/maybe
"""

# =========================================
# 2) Basic Helper Functions
# =========================================
def get_user_messages(slack_lines, user_name="Nassir Mohamud"):
    user_messages = []
    for line in slack_lines:
        if line.startswith(user_name):
            user_messages.append(line)
    return user_messages

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

def get_user_sentiment_score(user_messages):
    """Compute average sentiment score (1 for positive, -1 for negative)."""
    if not user_messages:
        return 0
    sentiment_sum = 0
    total_messages = 0
    for msg in user_messages:
        analysis = sentiment_pipeline(msg[:512])[0]
        label = analysis["label"]
        if label == "POSITIVE":
            sentiment_sum += 1
        elif label == "NEGATIVE":
            sentiment_sum -= 1
        total_messages += 1
    if total_messages == 0:
        return 0
    return sentiment_sum / total_messages

def get_user_engagement_rate(slack_lines, user_name="Nassir Mohamud"):
    """Count /going, /not-going, /maybe lines by that user, 
       then compute (commands typed / total lines)."""
    going_count = 0
    not_going_count = 0
    maybe_count = 0
    total_user_lines = 0

    for line in slack_lines:
        if line.startswith(user_name):
            total_user_lines += 1
            if "/going" in line:
                going_count += 1
            if "/not-going" in line:
                not_going_count += 1
            if "/maybe" in line:
                maybe_count += 1

    if total_user_lines == 0:
        return {"going": 0, "not_going": 0, "maybe": 0, "rate": 0.0}

    commands_typed = going_count + not_going_count + maybe_count
    engagement_rate = commands_typed / total_user_lines

    return {
        "going": going_count,
        "not_going": not_going_count,
        "maybe": maybe_count,
        "rate": engagement_rate
    }

def single_user_dropoff_chance(sentiment_score, engagement_info,
                               negative_sent_weight=0.5,
                               low_engagement_weight=0.5):
    """Heuristic: Weighted average of negative sentiment and (1 - engagement)."""
    sentiment_component = 0
    if sentiment_score < 0:
        sentiment_component = min(abs(sentiment_score) * 100, 100)
    user_rate = engagement_info["rate"]
    engagement_component = (1 - user_rate) * 100

    dropoff_chance = (negative_sent_weight * sentiment_component) + \
                     (low_engagement_weight * engagement_component)
    dropoff_chance = min(dropoff_chance, 100)
    return dropoff_chance


# =========================================
# 3) Streamlit App
# =========================================
st.title("Slack Drop-Off Analysis (Dynamic Username)")

# Split the slack text into lines once, at the start
lines = slack_text.strip().split('\n')

# Load the sentiment pipeline once
sentiment_pipeline = load_sentiment_pipeline()

# Create a text input for the username
user_input = st.text_input("Enter a Slack username:", "Nassir Mohamud")

# Button to run the analysis
if st.button("Analyze"):
    # 1) Gather messages for that user
    user_messages = get_user_messages(lines, user_name=user_input)

    # 2) Compute sentiment
    user_sentiment = get_user_sentiment_score(user_messages)

    # 3) Compute engagement
    user_engagement = get_user_engagement_rate(lines, user_name=user_input)

    # 4) Drop-Off
    user_dropoff = single_user_dropoff_chance(user_sentiment, user_engagement)

    # 5) Display results
    st.subheader(f"Results for {user_input}")
    st.write(f"**Average Sentiment Score**: {user_sentiment:.2f}")
    st.write(f"**Engagement Info**: {user_engagement}")
    st.write(f"**Estimated Drop-Off Chance**: {user_dropoff:.2f}%")

    # Explanation
    explanation = f"""
    **Estimated Drop-Off Chance** = {user_dropoff:.2f}%

    We are using a 50–50 weighting between **negative sentiment** and **(1 - engagement rate)**.
    - {user_input}'s sentiment is {"positive" if user_sentiment >= 0 else "negative"}, 
      so that contributes less or zero to drop-off if it's positive.
    - Engagement rate is {user_engagement["rate"]:.2f}, so from the engagement perspective,
      0 means a potential 100% drop-off.

    Hence, we combine those two factors for the final drop-off chance.
    """
    st.write("**Explanation**:")
    st.write(explanation)

st.write("---")
st.write("*Use the text box above to type any username from the Slack text. Then click 'Analyze'.*")
