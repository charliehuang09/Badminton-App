import os
import numpy as np
import pandas as pd
import cv2
from tqdm import trange, tqdm
import config
import argparse
def gaussian(x, mu, sig):
    return (
        np.exp(-np.power((x - mu) / (sig / 2), 2.0) / 2)
    )

def convert_y(input):
    meshgrid, _, _ = np.meshgrid(np.linspace(0, config.classes, config.classes + 1), np.linspace(0, 0, 640), np.linspace(0, 0, 360), indexing='ij')
    output = np.empty((config.classes + 1, 640, 360))
    for i in range(config.classes + 1):
        output[i] = (meshgrid[i] == input)
    return output


def get_y(visibility, x_coord, y_coord, std):
    x, y = np.indices([640, 360])
    if (visibility == 1 or visibility == 2 or visibility == 3):
        x_coord = round(x_coord / 2)
        y_coord = round(y_coord / 2)
        x = gaussian(x, x_coord, std)
        y = gaussian(y, y_coord, std)
    if (visibility == 0):
        x = np.zeros((640, 360))
        y = np.zeros((640, 360))
    # print(visibility)
    output = x * y
    # output *= (2 * np.pi * (std)**2) # seems to work if std is squared but not squared in paper (squared because gaussian generation is different?)
    output *= config.classes
    # print(output.shape, output.max())
    output = np.rint(output).astype(np.int16)
    return output

def get_x(imgs):
    # print(imgs[0].shape)
    # print(imgs[1].shape)
    # print(imgs[2].shape)
    a = cv2.resize(imgs[0], (640, 360)).swapaxes(0, 2)
    b = cv2.resize(imgs[1], (640, 360)).swapaxes(0, 2)
    c = cv2.resize(imgs[2], (640, 360)).swapaxes(0, 2)
    output = np.vstack((a, b, c)).astype(np.float32)
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
            y.append(get_y(df[i - 1][1], df[i - 1][2], df[i - 1][3], std=5))
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int16)
    return x, y

def Badmintonpreprocess(data_path, write_path):
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

def Tennispreprocess(data_path, write_path):
    index = 0
    for game in tqdm(os.listdir(data_path)):
        for clip in os.listdir(os.path.join(data_path, game)):
            q = []
            df = pd.read_csv(os.path.join(data_path, game, clip, 'Label.csv'))
            for i in range(len(os.listdir(os.path.join(data_path, game, clip))) - 1):
                q.append(cv2.imread(os.path.join(data_path, game, clip, f"{i:04d}.jpg")))

                if (len(q) == 3):
                    np.save(os.path.join(write_path, 'imgs', f"{str(index)}.npy"), get_x(q))
                    np.save(os.path.join(write_path, 'labels', f"{str(index)}.npy"), get_y(df.iloc[i]['visibility'], df.iloc[i]['x-coordinate'], df.iloc[i]['y-coordinate'], 5))
                    q.pop(0)
                    index += 1
    
    return
    
def clear(path):
    filelist = [ f for f in os.listdir(path) if f.endswith(".npy") ]
    for f in filelist:
        os.remove(os.path.join(path, f))
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='b=Badminton Dataset; t=Tennis Dataset')
    args = parser.parse_args()
    assert args.type == 'b' or args.type == 't', 'Wrong arguments'

    clear('data/train/imgs')
    clear('data/train/labels')
    clear('data/valid/imgs')
    clear('data/valid/labels')
    
    if (args.type == 'b'):
        Badmintonpreprocess('raw_data/train', 'data/train')
        Badmintonpreprocess('raw_data/valid', 'data/valid')
    if (args.type == 't'):
        Tennispreprocess('raw_data_tennis/train', 'data/train')
        Tennispreprocess('raw_data_tennis/valid', 'data/valid')

if __name__=='__main__':
    main()
