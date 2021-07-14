import pandas as pd
dfs = []
n = int(input())
for i in range(n):
	df = pd.read_csv(input()).df.dropna()
	
	df['content'] = df['content'].str.lower()\
	.replace({r"http\S+":""}, regex=True)\
	.replace({r"<.+>":" "},regex=True)\
	.replace({r"[ \n\r]+":" "},regex=True)\
	.replace({r"[^а-я0-9,.?!:\- \"\\()*+_]":""},regex=True)\

	df = df[df['content'].str.contains(r"[а-я]")]
	dfs.append(df)
pd.concat(dfs).to_csv("processed_data.csv")
	
