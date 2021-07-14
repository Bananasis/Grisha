import pandas as pd
import pytils.translit

dfs = []
n = int(input())
for i in range(n):
    df = pd.read_csv(input()).dropna()

    df['content'] = df['content'].str.lower() \
        .replace({r"http\S+": ""}, regex=True) \
        .replace({r"<.+>": " "}, regex=True) \
        .replace({r"[ \n\r]+": " "}, regex=True) \
        .replace({r"[^а-яa-z0-9,.?!:\- \"\\()*+_]": ""}, regex=True) \
        .map(pytils.translit.detranslify)

    df['author'] = df['author'].str.lower() \
        .replace({r"[^а-яa-z0-9]": ""}) \
        .map(pytils.translit.detranslify)

    df = df[df['content'].str.contains(r"[а-яa-z]")]

    dfs.append(df)
pd.concat(dfs).to_csv("processed_data.csv")
