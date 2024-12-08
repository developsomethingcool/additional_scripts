from rembg import remove
from PIL import Image, ImageOps
import io
import os

# Path to the dataset folder
dataset_dir = os.path.join("", "reals")
updated_dataset_dir = os.path.join("", "real_images")

def get_jpg_image_names(folder_path, format=".png"):
    """
    Gets the names of files with the specified format in the given folder.

    Args:
    folder_path (str): The path to the folder where the files are located.
    format (str): The file format to look for (default is ".jpg").

    Returns:
    list: A list of file names with the specified format.
    """
    # Check if the format is ".jpg"
    if format == ".png":
        # List comprehension to find all files ending with '.jpg' in the specified folder
        jpg_images = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.png')]
        # Return the list of jpg file names
        return jpg_images
    else:
        return

# Get the list of JPG images in the dataset folder
jpg_images = get_jpg_image_names(dataset_dir)

for image in jpg_images:
    input_path = dataset_dir + "/" + image
    output_path = updated_dataset_dir + "/" + image

    # Open the input image
    inp = Image.open(input_path)

    # Convert the image to byte data for rembg processing
    with io.BytesIO() as output_bytes:
        inp.save(output_bytes, format='PNG')
        image_bytes = output_bytes.getvalue()

    # Remove the background using rembg
    output_bytes = remove(image_bytes)

    # Load the output image into a Pillow Image object
    output = Image.open(io.BytesIO(output_bytes))

    # Create a white background image
    white_background = Image.new("RGB", output.size, (255, 255, 255))

    # Paste the output image onto the white background
    # The mask ensures that only non-transparent areas are pasted
    white_background.paste(output, mask=output if output.mode == 'RGBA' else None)

    # Save the final image with the white background
    white_background.save(output_path)