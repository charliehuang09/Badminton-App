import os
import numpy as np
import pandas as pd
import cv2
from tqdm import trange, tqdm
import config
from scipy.special import softmax
def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

def convert_y(input):
    meshgrid, _, _ = np.meshgrid(np.linspace(0, config.classes, config.classes + 1), np.linspace(0, 0, 640), np.linspace(0, 0, 360), indexing='ij')
    output = np.empty((config.classes + 1, 640, 360))
    for i in range(config.classes + 1):
        output[i] = (meshgrid[i] == input)
    return output


def get_y(visibility, x_coord, y_coord, std=5.0):
    assert visibility == 1
    x, y = np.indices([1280, 720])
    x = gaussian(x, x_coord, std)
    y = gaussian(y, y_coord, std)
    output = x * y
    output *= (2 * np.pi * (std)**2) # seems to work if std is squared but not squared in paper (squared because gaussian generation is different?)
    output *= config.classes + 1 #resize messes things up
    output = cv2.resize(output, (360, 640))
    output = np.rint(output).astype(np.int16)
    return output

def get_x(imgs):
    imgs[0] = cv2.resize(imgs[0], (640, 360)).swapaxes(0, 2)
    imgs[1] = cv2.resize(imgs[1], (640, 360)).swapaxes(0, 2)
    imgs[2] = cv2.resize(imgs[2], (640, 360)).swapaxes(0, 2)
    output = np.vstack((imgs[0], imgs[1], imgs[2]))
    return output
    

def extract_data(csv_path, video_path):
    x = []
    y = []
    df = pd.read_csv(csv_path).to_numpy()
    cap = cv2.VideoCapture(video_path)
    length = len(df)
    
    q = [np.zeros((720, 1280, 3)), np.zeros((720, 1280, 3)), np.zeros((720, 1280, 3))]
    for i in range(length):
        _, frame = cap.read()
        q.pop(0)
        q.append(frame)
        
        if (i % 10 == 0 and df[i - 1][1] == 1):
            x.append(get_x(q))
            y.append(get_y(df[i - 1][1], df[i - 1][2], df[i - 1][3]))
    x = np.array(x, dtype=np.float32) / 255
    y = np.array(y, dtype=np.int16)
    return x, y

def preprocess(data_path, write_path):
    csv_paths = []
    video_paths = []
    for match in os.listdir(data_path):
        for path in os.listdir(os.path.join(data_path, match, 'csv')):
            csv_paths.append(os.path.join(data_path, match, 'csv', path))
            video_paths.append(os.path.join(data_path, match, 'video', path.replace('_ball.csv', '.mp4')))
            
    index = 0
    for i in trange(len(csv_paths)):
        csv_path = csv_paths[i]
        video_path = video_paths[i]
        x_imgs, y_imgs = extract_data(csv_path, video_path)
        for x_img, y_img in zip(x_imgs, y_imgs):
            np.save(os.path.join(write_path, 'imgs', str(index)) + '.npy', x_img)
            np.save(os.path.join(write_path, 'labels', str(index)) + '.npy', y_img)
            index += 1
    
def clear(path):
    filelist = [ f for f in os.listdir(path) if f.endswith(".npy") ]
    for f in filelist:
        os.remove(os.path.join(path, f))
        
def main():
    clear('data/train/imgs')
    clear('data/train/labels')
    clear('data/valid/imgs')
    clear('data/valid/labels')
    
    preprocess('raw_data/train', 'data/train')
    preprocess('raw_data/valid', 'data/valid')

if __name__=='__main__':
    main()
