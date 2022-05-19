# import module
from pdf2image import convert_from_path

# Store Pdf with convert_from_path function
images = convert_from_path('git_german_switzerland.pdf')
 
for i in range(len(images)):
    images[i].save('page'+ str(i) +'.jpg', 'JPEG')