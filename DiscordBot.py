import discord


class DiscordBot(discord.Client):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

        @self.event
        async def on_message(message):
            self.handler = handler(message, self)


class MessageHandler:
    def __init__(self, commands):
        self.commands = commands

    def __call__(self, message, client):
        if message.author == client.user:
            return self
        elif message.content.startswith("_"):
            split = message.content.split()
            cmd = split[0].replace("_", "")
            argv = split[1:]
            if cmd not in self.commands:
                await message.channel.send("Unexpected command {}, try one of those {}".format(cmd, self.commands))
                return self
            new_handler = self.commands[cmd](argv, message, client)
            if not new_handler:
                return self
            return new_handler
        return self


class Command:
    def __init__(self, argn):
        self.argn = argn

    def __call__(self, argv, message, client):
        if len(argv) < self.argn:
            await message.channel.send("Not enough arguments expected  {}, got {}".format(self.argn, len(argv)))
        return None
