"""
Changes to psychopy PR655 class, in order to make it work correctly and more reliable.
"""
import time

from psychopy import logging
from psychopy.hardware import pr


class PR655(pr.PR655):
    """
    Class for Photometer PR655, adjusted for slower writing
    process.
    """

    def __init__(self, port):
        self.port = port
        super().__init__(self.port)
        self.com.timeout = None

    def slowWrite(self, message):
        """
        Send commands bit by bit.
        """
        if self.com.in_waiting == 0:
            for i in message:
                self.com.write(i.encode())
                time.sleep(0.1)

    def sendMessage(self, message, timeout=None, DEBUG=False):
        """
        Send a command to the photometer and wait an allotted
        timeout for a response (Timeout should be long for low
        light measurements).
        Override parent method for slower writing process, to
        assure reaction of photometer.

        :param message: Command to device.
        :param timeout: Length of waiting time for device to answer.
        :param DEBUG: If True, reports will be written to log.
        :return: Reply from device.
        """

        if message[-1] != '\n':
            message += '\n'  # append a newline if necessary

        # flush the read buffer first
        # read as many chars as are in the buffer
        self.com.read(self.com.inWaiting())

        # don't send bytes, as we encode then bit for bit!
        message = str(message)
        # send the message
        self.slowWrite(message)

        logging.debug(message)  # send complete message
        if message in ('d5\n', 'D5\n'):
            # we need a spectrum which will have multiple lines
            self.com.timeout = 3.
            reply = self.com.readlines()
            reply = [thisLine.decode().replace("\r\n", "").strip('"') for thisLine in reply]
            self.com.timeout = None
        else:
            reply = self.com.readline()
            reply = reply.decode().replace("\r\n", "").strip('"')

        self.com.flush()
        self.com.flushInput()
        self.com.flushOutput()

        return reply
