import os
import cv2
import params
import numpy as np
import pandas as pd

img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels

data_dir = params.data_dir
out_dir = params.out_dir
model_dir = params.model_dir


def preprocess(img, color_mode='RGB',brightness_mode="no_bright"):
    '''resize and crop the image
    :img: The image to be processed
    :return: Returns the processed image'''
# ##Chop and resize 
    # Chop off 1/2 from the top and cut bottom 150px(which contains the head of car)
    ratio = img_height / img_width
    h1, h2 = int(img.shape[0] / 2), img.shape[0] - 150
    w = (h2 - h1) / ratio
    padding = int(round((img.shape[1] - w) / 2))
    img = img[h1:h2, padding:-padding]
    # Resize the image
    img = cv2.resize(img, (img_width, img_height),interpolation=cv2.INTER_AREA)
    if color_mode == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Change Brightness
    if brightness_mode == 'bright':
        img = img_change_brightness(img)

    #Image Normalization
    #img = img_normalization(img) # memory exhausted
    return img

# ##Chop and resize 
#     ##Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
#     shape = img.shape
#     img = img[int(shape[0]/3):shape[0]-150, 0:shape[1]]
#     ## Resize the image
#     img = cv2.resize(img, (params.img_width, params.img_height), interpolation=cv2.INTER_AREA)

#     ## Return the image sized as a 4D array
#     return np.resize(img, (params.img_width, params.img_height, params.img_channels))


def frame_count_func(file_path):
    '''return frame count of this video'''
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count


def load_data(mode, color_mode='RGB', flip=True):
    '''get train and valid data,
    mode: train or valid, color_mode:RGB or YUV
    output: batch data.'''
    if mode == 'train':
        epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    elif mode == 'test':
        epochs = [10]
    else:
        print('Wrong mode input')

    imgs = []
    wheels = []
    # extract image and steering data
    for epoch_id in epochs:
        wheel_value = []

        vid_path = os.path.join(
            data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
        frame_count = frame_count_func(vid_path)
        cap = cv2.VideoCapture(vid_path)

        csv_path = os.path.join(
            data_dir, 'epoch{:0>2}_steering.csv'.format(epoch_id))
        rows = pd.read_csv(csv_path)
        wheel_value = rows['wheel'].values
        wheels.extend(wheel_value)

        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = preprocess(img, color_mode)
            imgs.append(img)

        assert len(imgs) == len(wheels)

        cap.release()
        
    if mode == 'train' and flip:
        augmented_imgs = []
        augmented_measurements = []
        for image, measurement in zip(imgs, wheels):
            augmented_imgs.append(image)
            augmented_measurements.append(measurement)
            # Flip images
            flipped_image = cv2.flip(image, 1)
            flipped_measurement = float(measurement) * -1.0
            augmented_imgs.append(flipped_image)
            augmented_measurements.append(flipped_measurement)

        X_train = np.array(augmented_imgs)
        y_train = np.array(augmented_measurements)
        y_train = np.reshape(y_train,(len(y_train),1))

    else:
        X_train = np.array(imgs)
        y_train = np.array(wheels)
        y_train = np.reshape(y_train,(len(y_train),1))

    return X_train, y_train


def load_batch(imgs, wheels):

    assert len(imgs) == len(wheels)
    n = len(imgs)

    assert n > 0

    batch_inbox = random.sample(range(0, n), params.batch_size)
    assert len(batch_inbox) == params.batch_size

    imgs_list, wheels_list = [], []
    for i in batch_inbox:
        imgs_list.append(imgs[i])
        wheels_list.append(wheels[i])

    return imgs_list, wheels_list


# Change Brightness
def img_change_brightness(img):
    """ Changing brightness of img to simulate day and night conditions    
    :param img: The image to be processed    
    :return: Returns the processed image    
   """    
    # Convert the image to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Compute a random brightness value and apply to the image
    brightness =  np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * brightness
    # Convert back to RGB and return
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)





# Normalization
# scale pixel value [0-255] to [0-1]
def img_normalization(img):
    img = img /255.0
    return img







## Add Shadow 

def add_shadow(img,orient='vertical'):
    h, w = img.shape[:2]

    if orient=='vertical':
        [x1, x2] = np.random.choice(w, 2, replace=False)
        k = h / (x2 - x1) # slope
        b = - k * x1      # y intercept 
        for i in range(h):
            c = int((i - b) / k)
            img[i, :c, :] = (img[i, :c, :] * .5) #.astype(np.int32)

    elif orient=='horizontal':
        [y1, y2] = np.random.choice(h, 2, replace=False)
        k = w / (y2 - y1) # slope
        b = - k * y1      # x intercept 
        for i in range(w):
            c = int((i - b) / k)
            img[:c, i, :] = (img[:c, i, :] * .5) #.astype(np.int32)

    return img


def add_shadow_images(idx, images, steerings, add_new=True):

    from math import ceil 
    rand_idx = np.random.choice(len(idx), ceil(len(idx)*0.2), replace=False)
    shadow_idx = idx[rand_idx]

    if add_new:
        
        idx_v = shadow_idx[:len(shadow_idx)//2]  
        idx_h = shadow_idx[len(shadow_idx)//2:]

        # add vertical shadow
        images_new_v = [add_shadow(img,'vertical') for img in images[idx_v]] # list
        images_new_v = np.stack(images_new_v, axis=0)   #numpy array
        steerings_new_v = steerings[idx_v]
        assert images_new_v.shape[1:]==(66,200,3)
        assert steerings_new_v.shape[0] == images_new_v.shape[0]
        
        # add horizontal shadow
        images_new_h = [add_shadow(img,'horizontal') for img in images[idx_h]] # list
        images_new_h = np.stack(images_new_h, axis=0)   #numpy array
        steerings_new_h = steerings[idx_h]
        assert images_new_h.shape[1:]==(66,200,3)
        assert steerings_new_h.shape[0] == images_new_h.shape[0]
        
        # combine
        images = np.concatenate((images, images_new_v,images_new_h), axis=0)
        steerings = np.concatenate((steerings,steerings_new_v,steerings_new_h), axis=0)
        assert images.shape[1:]==(66,200,3)
        assert steerings.shape[0] == images.shape[0]
        
        return images, steerings
    else:
        cnt = 0
        for i in shadow_idx:
            if cnt< (len(shadow_idx)//2):
                images[i] = add_shadow(images[i],'vertical')
                
            else:
                images[i] = add_shadow(images[i],'horizontal')
            cnt+=1
        assert images.shape[1:]==(66,200,3)
        assert steerings.shape[0] == images.shape[0]
        
        return images, steerings