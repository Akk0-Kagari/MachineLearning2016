import sys
from PIL import Image

imageName = sys.argv[1]
im=Image.open(imageName)
im.transpose(Image.ROTATE_180).show()