from PIL import Image
import os

# Function to calculate the mean size of images in a directory
def mean_image_size(directory):
    total_width = 0
    total_height = 0
    count = 0

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Adjust file extensions as needed
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    total_width += width
                    total_height += height
                    count += 1
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    if count == 0:
        print("No valid images found in the directory.")
        return None

    mean_width = total_width / count
    mean_height = total_height / count
    return mean_width, mean_height

# Replace 'path_to_images_folder' with the path to your images directory
images_directory_cool = 'dataset/cool'
images_directory_hot = 'dataset/hot'
mean_size_cool = mean_image_size(images_directory_cool)
mean_size_hot = mean_image_size(images_directory_hot)
if mean_size_cool:
    print(f"The mean size of images in the cool is {mean_size_cool[0]} x {mean_size_cool[1]} pixels.")

if mean_size_hot:
    print(f"The mean size of images in the hot is {mean_size_hot[0]} x {mean_size_hot[1]} pixels.")
