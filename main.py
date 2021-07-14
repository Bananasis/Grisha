import GenerateText
import DiscordBot
import ParseChanel

pc = ParseChanel.ParseChannel(limit=100000)
coms = {"scan": pc}
folder = input("Enter model folder (empty if none):")
if folder == "":
    gt = GenerateText.GenerateText(folder)
    coms = {"scan": pc, "talk": gt}

mh = DiscordBot.MessageHandler(coms)
db = DiscordBot.DiscordBot(mh)
db.run("put_token_here")
