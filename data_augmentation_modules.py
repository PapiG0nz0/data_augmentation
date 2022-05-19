from ast import Return
import os
from random import seed
from importlib_metadata import files
import tensorflow as tf
from PIL import Image
import numpy as np
import re
import time 
import albumentations as A



# img = Image.open('E:\Data Augmentation\Entrenamiento\A\A_0.jpg')
# output_path = 'E:\Data Augmentation\Entrenamiento\Output\\'
# input_path = 'E:\Data Augmentation\Entrenamiento\Test\\'
start_time = time.perf_counter()
str_template_time = None
class_name = 'Default'

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def  get_seed():
    seed1=tf.random.Generator.from_non_deterministic_state(1).normal([])
    seed2=tf.random.Generator.from_non_deterministic_state(1).normal([])
    seed = [seed1,seed2]
    return seed

def save_img(img,index,output_path,class_name):
    img_arr = np.array(img)
    save_img = Image.fromarray(img_arr)
    str_template = output_path+class_name+str(index)+'.jpg'
    save_img.save(str_template)
    return print('Imagen: '+class_name+str(index)+', guardada exitosamente')

def get_img_path(output_path):
    output_path = output_path
    return output_path
    
def random_brightness(input_path,output_path,class_name):
    start = time.time()
    count = sum(len(files) for _, _, files in os.walk(input_path))
    index = count
    for images in sorted_alphanumeric(os.listdir(input_path)):
            if(images.endswith(".jpg") or images.endswith(".png")):
                seed = get_seed()
                print(seed)
                img = Image.open(input_path+images)
                img_r_brightness = (tf.image.stateless_random_brightness(img, max_delta=.4, seed=seed))
                save_img(img_r_brightness, index, output_path, class_name)
                index = index + 1
    end = time.time()
    exec_time = end-start
    str_template_time = (str(exec_time)+" segundos en procesar #"+str(count)+" imagenes")
    return str_template_time

def random_contrast(input_path,output_path, class_name):
    start = time.time()
    count = sum(len(files) for _, _, files in os.walk(input_path))
    index = count
    for images in sorted_alphanumeric(os.listdir(input_path)):
            if(images.endswith(".jpg") or images.endswith(".png")):
                    seed = get_seed()
                    print(seed)
                    img = Image.open(input_path+images)
                    img_r_contrast = tf.image.stateless_random_contrast(img, lower = 0.3 , upper = 1, seed=seed)
                    save_img(img_r_contrast, index, output_path, class_name)
                    index = index + 1
                    str_template_time = "--- %s segundos en procesar #%i imagenes ---" % ((time.perf_counter() - start_time), count)
    end = time.time()
    exec_time = end-start
    str_template_time = (str(exec_time)+" segundos en procesar #"+str(count)+" imagenes")
    return str_template_time

def flip_left_right(input_path, output_path, class_name):
    start = time.time()
    count = sum(len(files) for _, _, files in os.walk(input_path))
    index = count
    for images in sorted_alphanumeric(os.listdir(input_path)):
            if(images.endswith(".jpg") or images.endswith(".png")):
                    seed = get_seed()
                    print(seed)
                    img = Image.open(input_path+images)
                    img_flip_left_right = tf.image.flip_left_right(img)
                    save_img(img_flip_left_right, index, output_path, class_name)
                    index = index + 1
                    str_template_time = "--- %s segundos en procesar #%i imagenes ---" % ((time.perf_counter() - start_time), count)
    end = time.time()
    exec_time = end-start
    str_template_time = (str(exec_time)+" segundos en procesar #"+str(count)+" imagenes")
    return str_template_time

def up_down(input_path):
    start = time.time()
    count = sum(len(files) for _, _, files in os.walk(input_path))
    index = count
    for images in sorted_alphanumeric(os.listdir(input_path)):
            if(images.endswith(".jpg") or images.endswith(".png")):
                    seed = get_seed()
                    print(seed)
                    img = Image.open(input_path+images)
                    img_flip_up_down = tf.image.flip_up_down(img)
                    save_img(img_flip_up_down, index, output_path)
                    index = index + 1
    end = time.time()
    exec_time = end-start
    str_template_time = (str(exec_time)+" segundos en procesar #"+str(count)+" imagenes")
    return str_template_time

def random_hue(input_path, output_path):
    start = time.time()
    count = sum(len(files) for _, _, files in os.walk(input_path))
    index = count
    for images in sorted_alphanumeric(os.listdir(input_path)):
            if(images.endswith(".jpg") or images.endswith(".png")):
                    seed = get_seed()
                    print(seed)
                    img = Image.open(input_path+images)
                    img_r_hue = tf.image.stateless_random_hue(img, max_delta=0.4, seed=seed)
                    save_img(img_r_hue, index, output_path)
                    index = index + 1
    end = time.time()
    exec_time = end-start
    str_template_time = (str(exec_time)+" segundos en procesar #"+str(count)+" imagenes")
    return str_template_time

def random_jpeg_quality(input_path):
    start = time.time()
    count = sum(len(files) for _, _, files in os.walk(input_path))
    index = count
    for images in sorted_alphanumeric(os.listdir(input_path)):
            if(images.endswith(".jpg") or images.endswith(".png")):
                    seed = get_seed()
                    print(seed)
                    img = Image.open(input_path+images)
                    img_r_jpeg_quality = tf.image.stateless_random_jpeg_quality(img, 10, 75, seed=seed)
                    save_img(img_r_jpeg_quality, index, output_path)
                    index = index + 1
    end = time.time()
    exec_time = end-start
    str_template_time = (str(exec_time)+" segundos en procesar #"+str(count)+" imagenes")
    return str_template_time

def random_saturation(input_path):
    start = time.time()
    count = sum(len(files) for _, _, files in os.walk(input_path))
    index = count
    for images in sorted_alphanumeric(os.listdir(input_path)):
            if(images.endswith(".jpg") or images.endswith(".png")):
                    seed = get_seed()
                    print(seed)
                    img = Image.open(input_path+images)
                    img_r_saturation = tf.image.stateless_random_saturation(img, lower= 0.1, upper=1, seed=seed)
                    save_img(img_r_saturation, index, output_path)
                    index = index + 1
    end = time.time()
    exec_time = end-start
    str_template_time = (str(exec_time)+" segundos en procesar #"+str(count)+" imagenes")
    return str_template_time

def total_random(input_path):
    start = time.time()
    transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ]) 
    count = sum(len(files) for _, _, files in os.walk(input_path))
    index = count
    for images in sorted_alphanumeric(os.listdir(input_path)):
            if(images.endswith(".jpg") or images.endswith(".png")):
                    seed = get_seed()
                    print(seed)
                    image = np.array(Image.open(input_path+images))
                    augmented_image = transform(image=image)['image']
                    save_img(augmented_image, index, output_path)
                    index = index + 1
    end = time.time()
    exec_time = end-start
    str_template_time = (str(exec_time)+" segundos en procesar #"+str(count)+" imagenes")
    return str_template_time

# random_brightness(input_path) 1 
# random_contrast(input_path) 2 
# flip_left_right(input_path) 3
# up_down(input_path) 4 
# random_hue(input_path) 5
# random_jpeg_quality(input_path) 6
# random_saturation(input_path) 7
# total_random(input_path) 8