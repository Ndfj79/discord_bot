import discord
from discord.ext import commands
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

intents = discord.Intents.all()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Привет! Я бот {bot.user}!')

@bot.command()
async def heh(ctx, count_heh=5):
    await ctx.send("he" * count_heh)

@bot.command()
async def check_image(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_url = attachment.url
            await attachment.save(f"./{attachment.filename}")
            # await ctx.send(f"Сохранили картинку в ./{attachment.filename}")

            np.set_printoptions(suppress=True)

            # Load the model
            model = load_model("keras_Model.h5", compile=False)

            # Load the labels
            class_names = open("labels.txt", "r").readlines()

            # Create the array of the right shape to feed into the keras model
            # The 'length' or number of images you can put into the array is
            # determined by the first position in the shape tuple, in this case 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Replace this with the path to your image
            image = Image.open(attachment.filename).convert("RGB")

            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predicts the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            # print("Class:", class_name[2:], end="")
            # print("Confidence Score:", confidence_score)

            if "white" in class_name[2:]:
                await ctx.send("Это белая одежда. Белые вещи можно стирать при температуре до 65 °C, для цветных изделий подходит температура от 30 до 40 °C (при стирке в горячей воде одежда может выцвести). Оптимальный режим отжима — до 800 оборотов в минуту. Цветные и белые вещи стирайте отдельно. Прежде чем положить одежду в машинку, проверьте, не линяет ли она.")
            elif 'black' in class_name[2:]:
                await ctx.send('Это чёрная одежда. Стирайте черную одежду только в холодной воде и выбирайте в настройках стиральной машинки деликатный режим. Чтобы сохранить черный цвет одежды, стирайте ее отдельно от одежды любых других цветов. Для сохранения яркости цвета при стирке добавляйте к стиральному порошку щепотку соли.')
            elif 'coloring' in class_name[2:]:
                await ctx.send('Это одежда с принтом. Прежде всего следует понимать, что принт является уязвимым местом в изделии, особенно, что касается стирки футболки. Стирать футболку с принтом можно, но стирать нужно на деликатном режиме стирки (макс. 600 оборотов при отжиме) наизнанку.')
    else:
        await ctx.send("Вы забыли загрузить картинку :(")


bot.run("")