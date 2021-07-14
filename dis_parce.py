import discord
import pandas as pd
import sys


client = discord.Client()
guild = discord.Guild


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    elif message.content.startswith("_"):

        cmd = message.content.split()[0].replace("_", "")
        if len(message.content.split()) > 1:
            parameters = message.content.split()[1:]


        if cmd == "scan":

            data = pd.DataFrame(columns=["content", "time", "author"])

            def is_command(msg):  # Checking if the message is a command call
                if len(msg.content) == 0:
                    return False
                elif msg.content.split()[0] == "_scan":
                    return True
                else:
                    return False

            async for msg in message.channel.history(
                limit=100000
            ):  # As an example, I've set the limit to 10000
                if msg.author != client.user:  # meaning it'll read 10000 messages instead of
                    if not is_command(msg):  # the default amount of 100
                        data = data.append(
                            {
                                "content": msg.content,
                                "time": msg.created_at,
                                "author": msg.author.name,
                            },
                            ignore_index=True,
                        )
                    if len(data) == 100000:
                        break

            file_location = (
                str(message.channel)+"_data.csv"  # Set the string to where you want the file to be saved to
            )
            data.to_csv(file_location)
print("begin")
client.run("ODYyNDMyNDk0NDAyMDExMTM2.YOYQ2Q.Nj5tiRCPXck92C_3yCGIzG7dz4s")
print("done")