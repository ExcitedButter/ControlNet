from PIL import Image
import os

def resize_images_in_folder(folder, base=64):
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png")):  # Add other file extensions if needed
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            w, h = img.size
            new_w = round(w / base) * base
            new_h = round(h / base) * base
            resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_img.save(img_path)  # This overwrites the original image, use a different path to keep originals
            img.close()


# Paths to the source and target directories
source_dir = './training/source'
target_dir = './training/target'

resize_images_in_folder(source_dir)
resize_images_in_folder(target_dir)



# from PIL import Image
# import os

# def resize_to_nearest_64(img, base=64):
#     w, h = img.size
#     new_w = round(w / base) * base
#     new_h = round(h / base) * base
#     return img.resize((new_w, new_h), Image.ANTIALIAS)

# def convert_and_delete_images_in_folder(folder):
#     for filename in os.listdir(folder):
#         if filename.endswith(".jpg"):
#             img_path = os.path.join(folder, filename)
#             img = Image.open(img_path)
#             img = img.convert('RGB')
#             img = resize_to_nearest_64(img)
#             new_filename = filename.replace(".jpg", ".png")
#             new_img_path = os.path.join(folder, new_filename)
#             img.save(new_img_path)
#             img.close()
#             os.remove(img_path)

# source_dir = './training/source'
# target_dir = './training/target'

# convert_and_delete_images_in_folder(source_dir)
# convert_and_delete_images_in_folder(target_dir)
