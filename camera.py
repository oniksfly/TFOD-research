import os
import cv2
import io
import numpy as np
import datetime
import subprocess as sp
from PIL import Image
from rawkit.raw import Raw

class Camera:
    def __init__(self):
        if not self.check_gphoto_installed():
            raise OSError()

    def check_gphoto_installed(self):
        result = False

        try:
            sp.run(["gphoto2", '--version'], stdout=sp.DEVNULL)
            result = True
        except FileNotFoundError:
            print("Seems gphoto2 isn't installed")

        return result

    def list_cameras(self): 
        result = []

        proc = sp.Popen(["gphoto2", "--summary"], stdout=sp.PIPE, stderr=sp.DEVNULL)
        output, _ = proc.communicate()
        rc = proc.returncode

        if rc == 0:
            result = [line.replace('Model:', '').strip() for line in output.decode("utf-8").split("\n") if 'Model' in line]

        return result

    def convert_file_name(original, new_name=None, new_extension=None):
        delimeter = original.rfind(".")
        original_name = ""
        original_extension = ""

        if delimeter < 0:
            original_name = original
        else:
            original_name = str(original[:delimeter])
            original_extension = str(original[delimeter+1:])

        if new_extension != None:
            original_extension = new_extension

        if new_extension != None:
            original_extension = new_extension

        if len(original_extension) > 0:
            return "{}.{}".format(original_name, original_extension)
        else:
            return original_name


    def capture_raw_file(self, name=None, path=None):
        proc = sp.Popen(["gphoto2", "--capture-image-and-download"], stdout=sp.PIPE, stderr=sp.DEVNULL)
        output, _ = proc.communicate()
        rc = proc.returncode

        if rc == 0:
            success_result = "Saving file as "

            messages = list(filter(lambda x: success_result in x, output.decode("utf-8").split("\n")))
            if len(messages) > 0:
                original_filename = messages[0].replace(success_result, '')
                new_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")

                if name != None:
                    new_name = name

                new_file_name = Camera.convert_file_name(original=original_filename, new_name=new_name)
                os.rename(original_filename, new_file_name)

                return new_file_name
            else:
                print("Can't get saved file name")

    def capture_compressed_file(self, name=None, path=None):
        raw_file = self.capture_raw_file(name=name, path=path)

        if raw_file != None:
            raw_image = Raw(raw_file)
            buffered_image = np.array(raw_image.to_buffer())
            image = Image.frombytes('RGB', (raw_image.metadata.width, raw_image.metadata.height), buffered_image)
            image.save(Camera.convert_file_name(raw_file, new_extension="png"), format='png')
    


i = Camera()
i.capture_compressed_file()

# paths = {
# }
                
# input_file = 'input_file_name.mp4'
# output_file = 'output_file_name.mp4'

# cap = cv2.VideoCapture('test.mp4')

# cv2.namedWindow('Butterfly' ,cv2.WINDOW_NORMAL)

# if (cap.isOpened()== False):
#     print("Error opening video stream or file")

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         cv2.imshow('Frame',frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#       break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()


#  gphoto2 --stdout --capture-movie | ffmpeg -re -i pipe:0 -listen 1 -f mjpeg http://localhost:8080/feed.jpg
# gphoto2 --stdout --capture-movie | ffmpeg -i - -pix_fmt yuv420p -threads 0 -f mp4 test.mp4