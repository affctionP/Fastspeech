
import telebot
import os
import json
from telebot import custom_filters, types
from telebot import types
from dotenv import load_dotenv
from telebot.states import State, StatesGroup
from telebot.storage import StateMemoryStorage
from telebot.types import ReplyParameters
import time, threading
from utils import *
from log_config import log_user_interaction ,app_logger,user_logger



# Initialize the bot
state_storage = StateMemoryStorage()
load_dotenv()
API_TOKEN = os.getenv('TELEGRAM_TOKEN')


bot = telebot.TeleBot(API_TOKEN,state_storage=state_storage)



@bot.message_handler(commands=['start'])
def send_welcome(message):

    log_user_interaction(message.from_user.id,message.from_user.username,"START BOT")
    welcom_message="سلام {} خوش اومدی به ربات"

    bot.reply_to(message,welcom_message.format(message.from_user.first_name))


@bot.message_handler()
def main_message_handler(message):
    chat_id = message.chat.id
    user_input = message.text
    
    with bot.retrieve_data(message.from_user.id, message.chat.id) as data:

        #data['converstion_data'].append("user :" +user_input)
        chat_create(message.from_user.id,"user :" +user_input+"\n" )
        response=get_answer(user_input)
        #data['converstion_data'].append("assistant:" +response)
        bot.send_message(chat_id, f"{response}")
        chat_create(message.from_user.id,"assistant:" +response+"\n")


   


def setup_filters():
    bot.add_custom_filter(custom_filters.StateFilter(bot))
    bot.add_custom_filter(custom_filters.TextMatchFilter())
    bot.add_custom_filter(custom_filters.TextStartsFilter())
    bot.add_custom_filter(custom_filters.IsDigitFilter())

# Function to start the bot
def start_bot():
    setup_filters()
    bot.infinity_polling()  # This will keep the bot running


start_bot()