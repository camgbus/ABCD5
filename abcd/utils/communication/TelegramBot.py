"""Telegram bot to pass messages about the training, or inform when experiments 
are done. Follow these links to get a token and chat-id
- https://www.christian-luetgens.de/homematic/telegram/botfather/Chat-Bot.htm
- https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id
Then, place these strings in a dictionary in the local directory.
"""

import telegram as tel
from abcd.local.telegram_logins import logins

class TelegramBot():
    def __init__(self, login_name='Default'):
        login_data = logins[login_name]
        self.chat_id = login_data['chat_id']
        self.bot = tel.Bot(token=login_data['token'])

    def send_msg(self, msg):
        self.bot.send_message(chat_id=self.chat_id, text=msg)
        