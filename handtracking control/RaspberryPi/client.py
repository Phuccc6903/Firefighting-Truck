import cv2
import requests
import numpy as np
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
import time


in1 = 13
in2 = 12
in3 = 21
in4 = 20
en1 = 6
en2 = 26


def forward(speed):
    setSpeedDefault(speed)
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)

def backward(speed):
    setSpeedDefault(speed)
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)


def turnright(speed):
    setSpeedDefault(speed)
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW)
    
def turnleft(speed):
    setSpeedDefault(speed)
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)


def stop():
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW) 

def setSpeedDefault(speed):
    p.start(speed)
    q.start(speed)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)
p=GPIO.PWM(en1,1000)
q=GPIO.PWM(en2,1000)

p.start(40)
q.start(40)

url = 'http://172.16.30.162:5000/prediction'

cam = cv2.VideoCapture(0)
while True:
    #if cv2.waitKey(1) & 0xFF == ord('r'):
    _,frame = cam.read()
    cv2.imshow('Camera Pi', frame)

    response = requests.get(url)
    if response.status_code == 200:
        prediction = response.text
        print("Received prediction:", prediction)
        if (prediction=='forward'):
            forward(45)
        elif (prediction=='right'):
            turnright(55)
        elif (prediction=='left'):
            turnleft(55)
        elif (prediction=='backward'):
            backward(45)
        else:
            stop()
    #time.sleep(0.5)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        stop()
        cam.release()
        break


                
stop()
cam.release()
cv2.destroyAllWindows()