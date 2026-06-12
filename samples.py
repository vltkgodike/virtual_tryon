from PIL import Image

img = Image.open(
    r"C:\Users\valkontek 010\Downloads\VTON-HD\train\openpose_img\14663_00_rendered.png"
)

print(img.size)