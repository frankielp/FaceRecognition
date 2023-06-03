from pymongo import MongoClient
from datetime import datetime
import pymongo
from PIL import Image
import io

# connect to db
uri = "mongodb+srv://npn279:grab2023@cluster0.ek6wvyn.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

# select database Bootcamp
db = client.Bootcamp

# select a collection (table) USER
collection = db.USER

document = collection.find_one({'_id': 'user3'})

image_binary = document['profile_image']

image_data = io.BytesIO(image_binary)
image = Image.open(image_data)

import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()
