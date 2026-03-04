from gpiozero import MotionSensor

from picamzero import Camera
import os

home_dir = os.environ['HOME'] #set location of home dir
cam = Camera()

pir = MotionSensor(21) #gpio pin 21 used
i = 0

while True:
    pir.wait_for_motion()
    print("You moved",i)
    cam.start_preview()
    cam.take_photo(f"{home_dir}/Desktop/motion test/new_image_{i}.jpg")
    cam.stop_preview()
    i+=1
    pir.wait_for_no_motion()