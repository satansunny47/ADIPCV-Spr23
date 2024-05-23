from PIL import Image

image_path = "flower.jpg"  
image = Image.open(image_path)

print("Image format:", image.format)
print("Image mode:", image.mode)
print("Image size:", image.size)

# Show the image
image.show()

