import Jetson.GPIO as GPIO


# Set up GPIO pins
GPIO.setmode(GPIO.BOARD)
input_pin = 7
GPIO.setup(input_pin, GPIO.IN) # reading input on this pin


"""
Expecting pull-down network with button connected to `input_pin`

    ^
    |      _____ button
    |     /
    |    /
    -- \----- input_pin
    |
    |
    |
    _
    -
"""

while True:

    print("Waiting for input...")

    # blocks function until rising edge detected
    GPIO.wait_for_edge(input_pin, GPIO.RISING) # wait for rising edge of input pin

    # only get here when a rising edge is seen on the input pin
    print("Rising edge detected!")

    