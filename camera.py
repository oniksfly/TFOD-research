import os
import shutil
import datetime
import subprocess as sp
from PIL import Image
import rawpy

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

        if new_name != None:
            original_name = new_name

        if new_extension != None:
            original_extension = new_extension

        if len(original_extension) > 0:
            return "{}.{}".format(original_name, original_extension)
        else:
            return original_name

    def set_shutter_speed_index(self, option_index: int) -> None:
        """
        Update camera's shutter speed value.
        As argument accepts key of `get_shutter_speed_options()`

        Parameters
        ----------
        option_index : int
            Index of property's value
        """

        if option_index not in self.get_shutter_speed_options().keys():
            raise ValueError("option_index is not available as property of `get_shutter_speed_options`")

        sp.Popen(["gphoto2", "--set-config-index", "/main/capturesettings/shutterspeed={}".format(option_index)], stdout=sp.PIPE, stderr=sp.DEVNULL)


    def get_shutter_speed_options(self) -> dict:
        """ 
        Get available shutterspeed options for current camera
        Key is available to set as camera's param option (index)
        Value is real-world value
        """
        result = {}
        proc = sp.Popen(["gphoto2", "--get-config", "shutterspeed"], stdout=sp.PIPE, stderr=sp.DEVNULL)
        output, _ = proc.communicate()
        rc = proc.returncode

        if rc == 0:
            for option in output.decode("utf-8").split("\n"):
                 if "Choice:" in option:
                     values = option.replace('Choice: ', '').split()
                     result[int(values[0])] = values[1]
        
        return result

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

                if path == None:
                    os.rename(original_filename, new_file_name)
                else:
                    new_path = os.path.join(path, new_file_name)
                    shutil.move(original_filename, new_path)
                    return new_path

                return new_file_name
            else:
                print("Can't get saved file name")

    def capture_compressed_file(self, name=None, path=None, keep_raw=False, show_photo=False) -> str:
        """
        Get compressed photo from camera
        
        Parameters
        ----------
        name : str
            Save image with specified name. If None provided save with timestamp (Default None)
        path : str
            Save image in specified location. If None provided save in current directory (Default None)
        keep_raw: bool
            Store raw and comressed file. (Default False)
        show_photo: bool
            Display photo after shoot. (Default False)
        """
        raw_file_name = self.capture_raw_file(name=name, path=path)

        if raw_file_name != None:
            compressed_file_name = Camera.convert_file_name(raw_file_name, new_extension="jpg")

            raw_image = rawpy.imread(raw_file_name)
            rgb_image = raw_image.postprocess(use_camera_wb=True)
            image = Image.fromarray(rgb_image)
            image.save(compressed_file_name)
            raw_image.close()

            if not keep_raw:
                os.remove(raw_file_name)

            if show_photo:
                self.show_photo(compressed_file_name)

            return compressed_file_name

    def show_photo(self, img_path: str) -> None:
        """
        Output image on screen

        Parameters
        ----------
        img_path : str
            Image name or path to show
        """
        image = Image.open(img_path)
        image.show()

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