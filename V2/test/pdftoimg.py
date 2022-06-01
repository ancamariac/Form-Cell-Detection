# import module
from pdf2image import convert_from_path

# Store Pdf with convert_from_path function
name = 'VAS'
images = convert_from_path(name + '.pdf')
 
for i in range(len(images)):
    images[i].save(name + str(i) +'.jpg', 'JPEG')