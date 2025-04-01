import os
import logging
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
import time
dp = Dispatcher()
import aiohttp

from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO)

@dp.message(Command("start"))
async def start_handler(message: Message):
    """
    Приветствие пользователя при вводе команды /start.
    """
    await message.reply("Привет! Отправь мне изображение с математической формулой, и я преобразую его в LaTeX-код.")

@dp.message(lambda message: message.photo is not None)
async def photo_handler(message: Message):
    """
    Обработка изображений с фотографией и отправка на сервер для распознавания формулы.
    """
    file_id = message.photo[-1].file_id
    file_info = await message.bot.get_file(file_id)
    file_path = file_info.file_path
    temp_filename = f"temp_{file_id}.jpg"
    await message.bot.download_file(file_path, temp_filename)

    latex_code = None

    # Отправка запроса на сервер
    async with aiohttp.ClientSession() as session:
        with open(temp_filename, 'rb') as f:
            form = aiohttp.FormData()
            form.add_field("image", f, filename=temp_filename, content_type='image/jpeg')
            try:
                async with session.post("http://im2latex.ru/convert-to-latex", data=form) as resp:
                    if resp.status == 200:
                        latex_code = await resp.text()
                    else:
                        logging.error(f"Ошибка ответа сервера: {resp.status}")
            except Exception as e:
                logging.error(f"Ошибка при запросе к серверу: {e}")

    if latex_code:
        await message.reply(f"Вот LaTeX-код твоей формулы:\n{latex_code[:170]}")
    else:
        await message.reply("Не удалось распознать формулу. Попробуй снова или проверь качество изображения.")

    try:
        os.remove(temp_filename)
    except Exception as e:
        logging.error(f"Ошибка при удалении файла {temp_filename}: {e}")
@dp.message()
async def default_handler(message: Message):
    """
    Обработка остальных сообщений.
    """
    await message.reply("Пожалуйста, отправь изображение с математической формулой.")


class TGClient:
    def __init__(self):
        TG_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        if not TG_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in env!")
        self.bot = Bot(token=TG_BOT_TOKEN)


    async def start(self):
        try:
            await dp.start_polling(self.bot)
        except Exception as e:
            logging.error(f"Ошибка запуска бота: {e}")
