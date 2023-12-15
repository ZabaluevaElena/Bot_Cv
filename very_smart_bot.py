import os
import io
import telebot
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

TRESHOLD = 0.5
BOT_TOKEN = os.environ.get('BOT_TOKEN')
START_MESSAGE = """Это очень умный бот:
он может отличить Чихуа-хуа от маффина с точностью >99%!
Присылайте фото и смотрите результат!"""

reconstructed_model = load_model("muffinvschihuahua.keras")

bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start', 'hello', 'help'])
def send_welcome(message):
    with open('muffin-meme2.jpg', 'rb') as img:
        bot.send_photo(message.chat.id, img, caption=START_MESSAGE)
    
@bot.message_handler(content_types=['photo'])
def photo(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    image = Image.open(io.BytesIO(downloaded_file))
    image = image.resize((224, 224))
    array = np.array(image)[np.newaxis,:,:,:]
    p_muff = reconstructed_model.predict(array)[0][0]
    if p_muff > TRESHOLD:
        bot.send_message(message.chat.id, f"Это маффин с вероятностью {p_muff:.2%}")
    else:
        bot.send_message(message.chat.id, f"Это Чихуа-Хуа с вероятностью {1-p_muff:.2%}")

bot.infinity_polling()
        
        
    
    
    
    

