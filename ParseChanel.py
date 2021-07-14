import pandas as pd
import DiscordBot as db


class ParseChannel(db.Command):
    def __init__(self, argn=1, limit=10000):
        self.limit = limit
        db.Command.__init__(self, argn)
        self.data = pd.DataFrame(columns=["content", "author"])

    def __call__(self, argv, message, client):
        db.Command.__call__(argv, message, client)
        channel = message.channel
        if len(argv) < self.argn:
            return None
        async for msg in channel.history(
                limit=self.limit
        ):
            if msg.author != client.user:
                if not msg.content.startswith("_"):
                    self.data = self.data.append(
                        {
                            "content": msg.content,
                            "author": msg.author.name,
                        },
                        ignore_index=True,
                    )
                if len(self.data) == self.limit:
                    break

        file_location = str(channel) + "_data.csv"

        self.data.to_csv(file_location)
