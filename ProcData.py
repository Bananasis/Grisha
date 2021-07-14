import pandas as pd
import pytils.translit

dfs = []
n = int(input())  # Number of files to process
for i in range(n):
    df = pd.read_csv(input()).dropna()

    df['content'] = df['content'].str.lower() \
        .replace({r"http\S+": ""}, regex=True) \
        .replace({r"<.+>": " "}, regex=True) \
        .replace({r"[ \n\r]+": " "}, regex=True) \
        .replace({r"[^а-яa-z0-9,.?!:\- \"\\()*+_]": ""}, regex=True) \
        .map(pytils.translit.detranslify)
    '''
    Replacing unwanted elements with regex
    str.lower() - make all symbols low register
    replace({r"http\S+": ""}) - remowe http(s) links
    replace({r"<.+>": " "}) - remowe emojis
    replace({r"[ \n\r]+": " "}) - remowe multiple white symbols
    replace({r"[^а-яa-z0-9,.?!:\- \"\\()*+_]": ""}) - remowe all symbols except wthitelist
    map(pytils.translit.detranslify) - cast from latin symbols to cyrilic 
    '''

    df['author'] = df['author'].str.lower() \
        .replace({r"[^а-яa-z0-9]": ""}) \
        .map(pytils.translit.detranslify)

    df = df[df['content'].str.contains(r"[а-я]")]  # Remove records with no text

    dfs.append(df)
pd.concat(dfs).to_csv("processed_data.csv")  # Join all processed data and save it to csv file
