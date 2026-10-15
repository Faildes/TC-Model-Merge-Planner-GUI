ratios = {
    "Portrait (2:3)": 2 / 3,
    "Standard (3:4)": 3 / 4,
    "Large Format (4:5)": 4 / 5, # 
    "Selfie, Social Media Videos (9:16)": 16 / 9,
    "jj": 13/19,
    "res": 9/7,
    "Square (1:1)": 1 / 1,
    "SD TV (4:3)": 1.33 / 1,
    "IMAX (1.43:1)": 1.43/1,
    "European Widescreen (1.66:1)": 1.66/1,
    "Widescreen / HD TV (16:9)": 16 / 9,
    "Standard Widescreen (1.85:1)": 1.85/1,
    "Cinemascope / Panavision (2.35:1)": 2.35/1,
    "Anamorphic Widescreen (2.39:1)": 2.39/1,
    "Older TV and some documentaries (4:3)": 4/3,
    "Golden Ratio (1.618:1)": 1.618 / 1,
    "Extended Format (4:7)": 1.75 / 1,
    "Image for Thumbnail": 800 / 450,
    "kk": 850 / 400,
    "h": 5/8,
}

target_pixels = 1638400
center_point = 1280

for name, ratio in ratios.items():
    width = int(center_point * (ratio**0.5))
    height = int(target_pixels / width)
    width = width - (width % 16)
    height = height - (height % 16)
    width_ex = int(center_point * (ratio**0.5)) // 16 * 16
    height_ex = int(target_pixels / width_ex) // 16 * 16
    total_pixels = width * height
    print(f"{name}, {width}x{height} (Exact: {width_ex}x{height_ex})")
