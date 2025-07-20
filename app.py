import streamlit as st
import pandas as pd
from  openai import OpenAI
import plotly.express as px



st.title(" 😍😍 Customer Review Sentiment Analyzer")
st.markdown(" This app Analyzes the sentiment of customer reviews to gain inights into thier opinions.")

user_input = st.text_input("Enter a customer review")
st.write("The current customer reivew is : ", user_input)


#import csv file reviews
df = pd.read_csv("reviews.csv")
st.write(df)





# OpenAI API Key input


openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key", 
    # api_key="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"

    type="password", 
    help="You can find your API key at https://platform.openai.com/account/api-keys"
)


def classify_sentiment_openai(review_text):
    """
    Classify the sentiment of a customer review using OpenAI's GPT-4o model.
    Parameters:
        review_text (str): The customer review text to be classified.
    Returns:
        str: The sentiment classification of the review as a single word, "positive", "negative", or "neutral".
    """
    client = OpenAI(api_key=openai_api_key)

    


    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 

    return completion.choices[0].message.content



# example usage
# st.title("Sentiment")
# st.write(classify_sentiment_openai(user_input))

# csv file uploader 

uploaded_file = st.file_uploader(
    "please upload  CSV file with restaurant reviews",
    type=("csv")

)

# once the user uploaded a CSV file
if uploaded_file is not None:
    # read the file
    reviews_df = pd.read_csv(uploaded_file)

# check if the data has a text column 
    text_columns = reviews_df.select_dtypes(include="object").columns

    if len(text_columns) == 0:
        st.error("No Text found")

# show dropdown menu to select the review column
    review_columns = st.selectbox(
    "Select the column with customer reivews",
        text_columns
)
    # Analyse the statement of the selected column

reviews_df["sentimentKhalid"] = reviews_df[review_columns].apply(classify_sentiment_openai)


# Display the sentiment distribution in metrics of 3 columns: positive, negative, and neutral

# make the strings in the sentiment column title
reviews_df["sentimentKhalid"] = reviews_df["sentimentKhalid"].str.title()
sentiment_counts = reviews_df["sentimentKhalid"].value_counts()
# st.write(reviews_df)
# st.write(sentiment_counts)

# create 3 columns to display 3 metrics
col1, col2, col3 = st.columns(3)

with col1:
    # show the number positive reviews and the percentage
    positive_count = sentiment_counts.get("Positive", 0)
    st.metric("Positive",
              positive_count,
              f"{positive_count / len(reviews_df) * 100:.2f}%")
    

with col2:
    # show the number neutral reviews and the percentage
    neutral_count = sentiment_counts.get("Neutral", 0)
    st.metric("Neutral",
              neutral_count,
              f"{neutral_count / len(reviews_df) * 100:.2f}%")
    


with col3:
    # show the number Negative reviews and the percentage
    negative_count = sentiment_counts.get("Negative", 0)
    st.metric("Negative",
              negative_count,
              f"{negative_count / len(reviews_df) * 100:.2f}%")
    
# Display the result the pie chart


fig = px.pie(
    values=sentiment_counts.values,
    names= sentiment_counts.index,
    title="Sentiment Distribution"
)
st.plotly_chart(fig)
# pushed all to the github to the new branch of main
