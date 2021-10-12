import glob

from PIL import Image, ImageDraw

fnames = [
    x.split("/")[-1].split(".")[0] for x in glob.glob("./SCUT-enstext/train/all_images/*.jpg")
]

for fname in fnames:
    image = Image.open(f"./SCUT-enstext/train/all_images/{fname}.jpg")
    mask = Image.new("RGB", image.size, color=(0, 0, 0))
    draw = ImageDraw.Draw(mask)

    with open(f"./SCUT-enstext/train/all_gts/{fname}.txt", "r") as file:
        lst = [[int(y) for y in x.split(",")] for x in file.read().splitlines()]

    for polygon in lst:
        temp = []
        for i in range(0, len(polygon), 2):
            x_cor, y_cor = polygon[i], polygon[i + 1]
            temp.append((x_cor, y_cor))
        draw.polygon(temp, fill=(255, 255, 255))

    mask.save(f"./SCUT-enstext/train/mask/{fname}.jpg")
