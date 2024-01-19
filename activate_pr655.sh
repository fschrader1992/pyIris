# Use with "sudo" under Linux

# Replace with vendor and product ID
sh -c "echo XXXX XXXX >/sys/bus/usb-serial/drivers/generic/new_id"
modprobe usbserial
chmod 0666 /dev/ttyUSB0
