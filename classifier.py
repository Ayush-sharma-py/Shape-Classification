import os
import matplotlib.image as img
import numpy
from tensorflow import keras
from scipy import misc
from PIL import Image,ImageDraw
images = os.listdir("C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Train")
resized_images = os.listdir("C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Resized")
train = os.listdir("C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Test")
resized_train = os.listdir("C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Resized_Test")
shapes = ["triangle","square","pentagon","hexagon"]
processsed_images = []




for i in images:
    ig = Image.open(f"C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Train\\{i}")
    ig = ig.resize((28,28))
    ig.save(f"C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Resized\\{i}")
for i in resized_images:
    n = img.imread(f"C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Resized\\{i}")
    processsed_images.append(n)

Test_images = numpy.array(processsed_images)

s = 0
for i in Test_images:
    for k in i:
        for x in k:
            noofshapes = len(x)
            for m in x:
                s += 1

div = s
Test_images = Test_images/255



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 3)),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(len(shapes))
])

label = []
with open("label.txt",'r') as f:
    data = f.read()
    label = data.split(",")
    f.close()

for i in range(0,len(label)):
    label[i] = float(label[i])

label = numpy.array(label)

model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#model.fit(Test_images, label, epochs=10,callbacks=cp_callback)

test_images = []
test_images_directory = []

for i in train:
    ig = Image.open(f"C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Test\\{i}")
    ig = ig.resize((28,28))
    ig.save(f"C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Resized_Test\\{i}")
for i in resized_train:
    n = img.imread(f"C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Resized_Test\\{i}")
    test_images_directory.append(f"C:\\Users\\Ayush Sharma\\Desktop\\CodeField\\Resized_Test\\{i}")
    test_images.append(n)


test_images_array = numpy.array(test_images)
probability_model = keras.Sequential([model, keras.layers.Softmax()])


predictions = probability_model.predict(test_images_array)
print(predictions)

import math
for i in range(0,len(predictions)):
    image = Image.open(test_images_directory[i])
    image = image.resize((500,500))
    for k in range(0,len(predictions[i])):
        if predictions[i][k]*100 > 25: 
            v = str(shapes[k]) + ":" + str(predictions[i][k]*100)

            try:
                x = round(math.sqrt(s))/noofshapes
            except:
                x = round(math.sqrt(s))
            noofshapes -= 1
            tx = ImageDraw.Draw(image)
            tx.text((x,x),v,fill=(0,0,0,0))
            
            image.save("a.png")
    image.show()

