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
# 2) Extract This User's Lines
# =========================================
def get_user_messages(slack_lines, user_name="Nassir Mohamud"):
    user_messages = []
    for line in slack_lines:
        if line.startswith(user_name):
            user_messages.append(line)
    return user_messages

lines = slack_text.strip().split('\n')
nassir_messages = get_user_messages(lines, user_name="Nassir Mohamud")


# =========================================
# 3) Sentiment Analysis (Hugging Face)
# =========================================
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_sentiment_pipeline()

def get_user_sentiment_score(user_messages):
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

nassir_sentiment_score = get_user_sentiment_score(nassir_messages)


# =========================================
# 4) Engagement Check
# =========================================
def get_user_engagement_rate(slack_lines, user_name="Nassir Mohamud"):
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

nassir_engagement = get_user_engagement_rate(lines, "Nassir Mohamud")


# =========================================
# 5) Drop-Off Heuristic
# =========================================
def single_user_dropoff_chance(
    sentiment_score,
    engagement_info,
    negative_sent_weight=0.5,
    low_engagement_weight=0.5
):
    # Sentiment
    sentiment_component = 0
    if sentiment_score < 0:
        sentiment_component = min(abs(sentiment_score) * 100, 100)

    # Engagement
    user_rate = engagement_info["rate"]
    engagement_component = (1 - user_rate) * 100

    # Weighted average
    dropoff_chance = (negative_sent_weight * sentiment_component) + \
                     (low_engagement_weight * engagement_component)
    # Cap at 100
    dropoff_chance = min(dropoff_chance, 100)
    return dropoff_chance

nassir_dropoff = single_user_dropoff_chance(nassir_sentiment_score, nassir_engagement)


# =========================================
# 6) Streamlit App UI
# =========================================
st.title("Slack Drop-Off Analysis")

user_name = "Nassir Mohamud"
st.subheader(f"Results for {user_name}")

st.write(f"**Average Sentiment Score**: {nassir_sentiment_score:.2f}")
st.write(f"**Engagement Info**: {nassir_engagement}")
st.write(f"**Estimated Drop-Off Chance**: {nassir_dropoff:.2f}%")

# Explanation
manual_explanation = f"""
**Estimated Drop-Off Chance** = {nassir_dropoff:.2f}%

We are using a 50–50 weighting between **negative sentiment** and **(1 - engagement rate)**.
- {user_name}'s sentiment is {"positive" if nassir_sentiment_score >= 0 else "negative"}, so that contributes less or zero to drop-off if it's positive.
- Engagement rate is {nassir_engagement["rate"]:.2f}, so from the engagement perspective, 0 means a potential 100% drop-off.

Hence, we combine those two factors for the final drop-off chance.
"""

st.write("**Explanation**:")
st.write(manual_explanation)

st.write("---")
st.write("*Thank you for using the app!*")

