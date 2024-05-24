import cv2 as cv
import numpy as np
import argparse
import os
from pathlib import Path

# Progress bar
from tqdm.contrib.itertools import product
# from itertools import product

import matplotlib.pyplot as plt
import time

def clear_dir(path):
    files = os.listdir(path)
    for file_name in files:
        file_path = os.path.join(path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Error: {e}")


def SSD(window1: np.array, window2: np.array):
    return np.sum((window1 - window2)**2)

#ZSSD. Assume que window1 já tem média subtraída
def robustSSD(window1: np.array, window2: np.array):
    return np.sum((window1 - (window2 - window2.mean()))**2)

# Passing args
if __name__ == '__main__':
    clear_dir('./disparities')
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")


    parser = argparse.ArgumentParser(
        prog='Calculate Disparities',
        description='Calcula matriz de disparidades entre duas images'
    )

    parser.add_argument('left_img')
    parser.add_argument('right_img')
    parser.add_argument('-s', '--search_range', type=int, default=55)
    parser.add_argument('-w', '--window', type=int, default=3)
    parser.add_argument('-r', '--robust', action='store_true')
    parser.add_argument('-ui', '--updateinterval', type=int, default=1000)

    args = parser.parse_args()
    args_d = vars(args)
    print("Parameters:\n")
    for arg in args_d:
        print(f'  {arg} = {args_d[arg]}')

    # Images
    img = Path(args.left_img).stem + '-' + Path(args.right_img).stem 
    img_l_name = args.left_img
    img_r_name = args.right_img

    # Window size
    WINDOW_SIZE = args.window
    kernel_half = max(0, WINDOW_SIZE // 2)
    # 
    search_range = args.search_range

    # Função de erro
    if args.robust:
        error_function = robustSSD
        error_function_name  = 'ZSSD'
    else:
        error_function = SSD
        error_function_name  = 'SSD' 

    # Progress bar update interval 
    updateinterval = args.updateinterval

    # Read imgs
    img_l = cv.imread(img_l_name)
    img_r = cv.imread(img_r_name)

    # Img Dimensions
    height, width = img_l.shape[0], img_l.shape[1]

    # Converts to CIELAB color space
    img_l = cv.cvtColor(img_l, cv.COLOR_BGR2LAB)
    img_r = cv.cvtColor(img_r, cv.COLOR_BGR2LAB)

    # Create the disparities matrix
    disparities = np.zeros((height, width))

    # For each pixel in the left image get the neighbors window and calculate the error
    for i, j in product(range(kernel_half, height - kernel_half), range(kernel_half, width - kernel_half), miniters=updateinterval):
        # Janelas
        window_l = img_l[
            i - kernel_half : i + kernel_half + 1,
            j - kernel_half : j + kernel_half + 1
        ]

        if args.robust:
            window_l = window_l - window_l.mean()

        min_val = 99999999
        disp = 0

        window_y1 = i - kernel_half
        window_y2 = i + kernel_half + 1
        window_x1 = j - kernel_half
        window_x2 = j + kernel_half + 1

        maximum_offset = min(search_range, window_x1)

        for offset in range(maximum_offset):
            window_r = img_r[
                window_y1 : window_y2,
                window_x1 - offset : window_x2 - offset
            ]
            val = error_function(window_l, window_r)
            if min_val > val:
                min_val = val
                disp = offset
        disparities[i, j] = disp

    # Compute Global cost

    plt.style.use('grayscale')
    plt.matshow(disparities)

    # Save the matrix
    out_file = 'disparities/' + error_function_name + '-' + f'{WINDOW_SIZE}x{WINDOW_SIZE}-' + f'd{search_range}' + img + '-' + timestamp + f'-{width}x{height}'
    np_file = np.save(out_file, disparities)

    # Save the matrix img
    cv.normalize(disparities, disparities, 1.0, 0.0, cv.NORM_INF)
    disparities = (disparities * 255).astype(int)

    cv.imwrite(out_file+'.png', disparities)

    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()