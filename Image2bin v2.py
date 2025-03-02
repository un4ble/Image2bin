
# MADE BY UN4BLE

# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License

# Github: un4ble

# --------------------------
#  Image to .bin converter 
# --------------------------

# Colab Compatibile version (run this script on Google Colab!)

# check Image2bin github repo for more info (https://github.com/un4ble/Image2bin)


import os
import struct
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

Image.MAX_IMAGE_PIXELS = None

output_dir = "/content/filename" # replace filename with name of your input image
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "result.bin")

im = Image.open("/content/filename") # replace filename with name of your input image
pix = np.array(im, dtype=np.uint8)

with open(output_file, 'wb') as outfile:
    outfile.write(struct.pack('<H', im.size[0]))
    outfile.write(struct.pack('<H', im.size[1]))
    print(f"Processing image {im.filename} with dimensions {im.size[0]}x{im.size[1]}")

    total_rows = im.size[1]
    progress_step = max(1, total_rows // 20)
    num_workers = min(cpu_count(), xx) # replace xx with number of cpu threads

    def process_row(y):
        row_data = ((pix[y, :, 0] >> 3) << 11) | ((pix[y, :, 1] >> 2) << 5) | (pix[y, :, 2] >> 3)
        return y, row_data.tobytes()

    with Pool(num_workers) as pool:
        for y, line in sorted(pool.imap_unordered(process_row, range(total_rows))):
            outfile.write(line)
            
            if y % progress_step == 0 or y == total_rows - 1:
                progress = (y / total_rows) * 100
                print(f"Processed {y}/{total_rows} rows ({progress:.1f}%)")

print("Processing complete! File saved as:", output_file)
