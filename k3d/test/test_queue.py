import unittest
from ..queue import Queue


class TestQueue(unittest.TestCase):
    def setUp(self):
        self.outputed = []
        self.queue = Queue(self.outputed.append)

    def test_adds_are_buffered(self):
        self.queue.add('x')

        self.assertFalse(self.outputWasCalled)

    def test_flush_outputs_buffered_adds(self):
        self.queue.add('x')
        self.queue.flush()

        self.assertTrue(self.outputWasCalled)

    def test_empty_buffer_flash(self):
        self.queue.flush()

        self.assertFalse(self.outputWasCalled)

    def test_adds_after_flush_are_not_buffered(self):
        self.queue.flush()
        self.queue.add('x')

        self.assertTrue(self.outputWasCalled)

    @property
    def outputWasCalled(self):
        return len(self.outputed) > 0
