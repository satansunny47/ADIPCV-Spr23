from PIL import Image

def saturate(image, saturation_factor):
    # Converting the image to the HSV color space
    image_hsv = image.convert("HSV")
    image_pixels = image_hsv.load()

    # Applying saturation factor to the saturation channel
    for y in range(image_hsv.size[1]):
        for x in range(image_hsv.size[0]):
            h, s, v = image_pixels[x, y]
            s = min(255, int(s * saturation_factor))
            image_pixels[x, y] = (h, s, v)

    # Converting back to RGB
    saturated_image = image_hsv.convert("RGB")
    return saturated_image

def desaturate(image, desaturation_factor):
    # Converting the image to the HSV color space
    image_hsv = image.convert("HSV")
    image_pixels = image_hsv.load()

    # Applying desaturation factor to the saturation channel
    for y in range(image_hsv.size[1]):
        for x in range(image_hsv.size[0]):
            h, s, v = image_pixels[x, y]
            s = max(0, int(s / desaturation_factor))
            image_pixels[x, y] = (h, s, v)

    # Converting back to RGB
    desaturated_image = image_hsv.convert("RGB")
    return desaturated_image

original_image_path = "flower.jpg"
original_image = Image.open(original_image_path)

# Defining saturation factors and operating 
saturation_factor = 2.0
desaturation_factor = 2.0

saturated_image = saturate(original_image, saturation_factor)
desaturated_image = desaturate(original_image, desaturation_factor)

saturated_image.save("maximally_saturated_image.jpg")
desaturated_image.save("desaturated_image.jpg")
# Get the sizes of the images for building the sat-desat image
width_saturated, height_saturated = saturated_image.size
width_desaturated, height_desaturated = desaturated_image.size

if height_saturated != height_desaturated:
    raise ValueError("Image heights do not match")

# Create a new image with combined width
combined_width = width_saturated + width_desaturated
combined_height = height_saturated
combined_image = Image.new("RGB", (combined_width, combined_height))

# Paste the saturated image and the desaturated image next to each other
combined_image.paste(saturated_image, (0, 0))
combined_image.paste(desaturated_image, (width_saturated, 0))

# Save the combined image
combined_image.save("saturated_desaturated_image.jpg")
