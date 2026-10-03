import gradio as gr
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("purchase_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_review(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("amazon_purchase_likelihood_balanced_2500.csv")

# Remove ID if present
if "id" in df.columns:
    df = df.drop("id", axis=1)


# ============================================================
# PREDICTION
# ============================================================

def predict_purchase(
    review,
    sentiment,
    category,
    price,
    discounted_price,
    reviews,
    rating,
    sentiment_score
):

    discount_amount = price - discounted_price

    if price > 0:
        discount_percentage = (discount_amount / price) * 100
    else:
        discount_percentage = 0

    review_length = len(review.split())

    new_data = pd.DataFrame({
        "clean_review": [clean_review(review)],
        "sentiment": [sentiment],
        "category": [category],
        "price": [price],
        "discounted_price": [discounted_price],
        "reviews": [reviews],
        "rating": [rating],
        "Discount Amount": [discount_amount],
        "Review Length": [review_length],
        "discount_percentage": [discount_percentage],
        "sentiment_score": [sentiment_score]
    })

    processed_data = preprocessor.transform(new_data)

    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    if prediction == 1:
        result = "🟢 LIKELY TO PURCHASE"
    else:
        result = "🔴 NOT LIKELY TO PURCHASE"

    details = (
        f"Purchase Probability: {probability:.2%}\n"
        f"Rating: {rating}/5\n"
        f"Discount: {discount_percentage:.1f}%"
    )

    return result, details


# ============================================================
# ANALYTICS
# ============================================================

def sentiment_chart():

    counts = df["sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(counts.index, counts.values)

    ax.set_title("Customer Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")

    for i, value in enumerate(counts.values):
        ax.text(i, value + 10, str(value), ha="center")

    plt.tight_layout()

    return fig


def purchase_chart():

    counts = df["purchase_likely"].value_counts()

    values = [
        counts.get(0, 0),
        counts.get(1, 0)
    ]

    labels = ["Not Likely", "Likely"]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(labels, values)

    ax.set_title("Purchase Likelihood Distribution")
    ax.set_xlabel("Purchase Likelihood")
    ax.set_ylabel("Customers")

    for i, value in enumerate(values):
        ax.text(i, value + 15, str(value), ha="center")

    plt.tight_layout()

    return fig


def rating_chart():

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(df["rating"], bins=10)

    ax.set_title("Product Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Products")

    plt.tight_layout()

    return fig


# ============================================================
# MODEL METRICS
# ============================================================

# Recreate test split for evaluation
from sklearn.model_selection import train_test_split

X = df[
    [
        "customer_review",
        "sentiment",
        "category",
        "price",
        "discounted_price",
        "reviews",
        "rating",
        "Discount Amount",
        "Review Length",
        "discount_percentage",
        "sentiment_score"
    ]
].copy()

X["clean_review"] = X["customer_review"].apply(clean_review)

X = X[
    [
        "clean_review",
        "sentiment",
        "category",
        "price",
        "discounted_price",
        "reviews",
        "rating",
        "Discount Amount",
        "Review Length",
        "discount_percentage",
        "sentiment_score"
    ]
]

y = df["purchase_likely"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_test_processed = preprocessor.transform(X_test)

y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)


# ============================================================
# UI
# ============================================================

categories = sorted(df["category"].unique())

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Purchase Decision",
    css="""
    .title {
        text-align: center;
        font-size: 34px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #666;
        margin-bottom: 20px;
    }

    .result {
        text-align: center;
        font-size: 25px;
        font-weight: bold;
    }
    """
) as dashboard:

    gr.Markdown(
        """
        <div class="title">
        🛒 Purchase Decision
        </div>

        <div class="subtitle">
        Machine Learning Dashboard for Customer Purchase Likelihood
        </div>
        """
    )

    # ========================================================
    # METRICS
    # ========================================================

    with gr.Row():

        gr.Markdown(
            f"""
            ### 📊 Dataset
            **{len(df):,} Records**
            """
        )

        gr.Markdown(
            f"""
            ### 🎯 Accuracy
            **{accuracy:.2%}**
            """
        )

        gr.Markdown(
            f"""
            ### 📈 ROC-AUC
            **{auc:.3f}**
            """
        )

        gr.Markdown(
            """
            ### 🤖 Model
            **Logistic Regression**
            """
        )

    # ========================================================
    # PREDICTION TAB
    # ========================================================

    with gr.Tab("🛒 Purchase Prediction"):

        gr.Markdown(
            """
            ## Predict Customer Purchase Likelihood

            Enter product and review information below.
            """
        )

        with gr.Row():

            with gr.Column():

                review = gr.Textbox(
                    label="Customer Review",
                    placeholder="Example: Excellent product, very good quality and worth the price.",
                    lines=5
                )

                sentiment = gr.Dropdown(
                    ["Positive", "Neutral", "Negative"],
                    value="Positive",
                    label="Sentiment"
                )

                category = gr.Dropdown(
                    categories,
                    value=categories[0],
                    label="Product Category"
                )

            with gr.Column():

                price = gr.Number(
                    value=2000,
                    label="Original Price (₹)"
                )

                discounted_price = gr.Number(
                    value=1500,
                    label="Discounted Price (₹)"
                )

                reviews = gr.Number(
                    value=1000,
                    label="Number of Reviews"
                )

                rating = gr.Slider(
                    1,
                    5,
                    value=4.5,
                    step=0.1,
                    label="Product Rating"
                )

                sentiment_score = gr.Slider(
                    -1,
                    1,
                    value=0.8,
                    step=0.01,
                    label="Sentiment Score"
                )

        predict_button = gr.Button(
            "🔮 Predict Purchase Likelihood",
            variant="primary"
        )

        result = gr.Textbox(
            label="Prediction",
            elem_classes="result"
        )

        details = gr.Textbox(
            label="Prediction Details"
        )

        predict_button.click(
            predict_purchase,
            inputs=[
                review,
                sentiment,
                category,
                price,
                discounted_price,
                reviews,
                rating,
                sentiment_score
            ],
            outputs=[
                result,
                details
            ]
        )

    # ========================================================
    # ANALYTICS
    # ========================================================

    with gr.Tab("📊 Analytics Dashboard"):

        gr.Markdown("## Customer & Product Analytics")

        with gr.Row():

            gr.Plot(
                sentiment_chart,
                label="Sentiment Analysis"
            )

            gr.Plot(
                purchase_chart,
                label="Purchase Likelihood"
            )

        gr.Plot(
            rating_chart,
            label="Rating Distribution"
        )

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    with gr.Tab("🤖 Model Performance"):

        gr.Markdown("## Machine Learning Performance")

        performance = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            "Score": [
                f"{accuracy:.2%}",
                f"{precision:.2%}",
                f"{recall:.2%}",
                f"{f1:.2%}",
                f"{auc:.3f}"
            ]
        })

        gr.Dataframe(
            performance,
            label="Model Evaluation"
        )

    # ========================================================
    # ABOUT
    # ========================================================

    with gr.Tab("ℹ️ About Project"):

        gr.Markdown(
            """
            ## About the Project

            **Project Title:**

            Impact of Product Ratings and Reviews on Customer
            Purchase Decisions Using Machine Learning

            ### Objective

            This project analyzes customer reviews, ratings,
            sentiment, pricing and other product attributes to
            predict customer purchase likelihood.

            ### Techniques Used

            - Data preprocessing
            - Text cleaning
            - TF-IDF
            - One-Hot Encoding
            - Logistic Regression
            - Random Forest
            - Confusion Matrix
            - ROC-AUC

            ### Dataset

            The project uses an Amazon-style synthetic dataset
            created for academic experimentation.
            """
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    dashboard.launch()