import config
from openai import OpenAI  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from embeddings import get_embedding, get_embedding_batch


client = OpenAI()  # defaults to getting the key using os.environ.get("OPENAI_API_KEY")


# load the amazon fine food reviews csv
embeddings_path = "embeddings/amazon_food_reviews_10.csv"
embedding_encoding = "cl100k_base"
max_tokens = 8000  # the maximum for text-embedding-3-small is 8191
# subsample to 1k most recent reviews and remove samples that are too long
top_n = 1000

embedding_model = "text-embedding-3-small"

# The index_col=0 parameter is used to designate the first column in the CSV as the index of the DataFrame, rather than adding an auto-incrementing integer index
df = pd.read_csv(embeddings_path, index_col=0)

# DataFrame is filtered to retain only the columns ["Time", "ProductId", "UserId", "Score", "Summary", "Text"]
df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]

# drop rows with missing values
df.dropna()

# combine the "Summary" and "Text" columns into a single "combined" column
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)

# Preview the Dataset
print(df.head(2))


# first cut to first 2k entries, assuming less than half will be filtered out
df = df.sort_values("Time").tail(top_n * 2)
df.drop("Time", axis=1, inplace=True)

encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)
print(len(df))

df["embedding"] = df.combined.apply(
    lambda x: get_embedding_batch(x, model=embedding_model)
)
df.to_csv("embeddings/amazon_food_reviews_10_with_embeddings.csv")
