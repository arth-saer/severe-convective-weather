import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def show_radar_image_size(image_path):
    """
    Displays the size of a radar image.

    Parameters:
    image_path (str): The file path to the radar image.
    """
    try:
        # Open the image using PIL
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Image Size: Width = {width}, Height = {height}")
            image_np = np.asarray(img)
            print(np.max(image_np))
            print(np.min(image_np))
    except Exception as e:
        print(f"Error loading image: {e}")
if __name__ == "__main__":
    radar_image_path = "/mnt/md1/ConvectionAirport/Datasets/Radar/radar_meteo_sz/2017/06/02/201706020206.png"  # Replace with your radar image path
    show_radar_image_size(radar_image_path)
