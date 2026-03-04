from picamzero import Camera
import os

home_dir = os.environ['HOME'] #set location of home dir
cam = Camera()

cam.start_preview()
cam.take_photo(f"{home_dir}/Desktop/new_image.jpg") # save img to desktop
cam.stop_preview()
#working as of 28/2/26
#install picamzero library via command line
#upon imaging new os, "sudo apt-get install -f" then "sudo dpkg --configure -a" to download
#then sudo apt update
#and sudo apt install python3-picamzero