import unittest
from objects import Objects

class TestObjects(unittest.TestCase):
    def setUp(self):
        self.outputed = []
        self.objects = Objects(lambda x: self.outputed.append(x))

    def test_adds_are_buffered(self):
        self.objects.add('x')

        self.assertFalse(self.outputWasCalled)

    def test_flush_outputs_buffered_adds(self):
        self.objects.add('x')
        self.objects.flush()

        self.assertTrue(self.outputWasCalled)

    def test_empty_buffer_flash(self):
        self.objects.flush()

        self.assertFalse(self.outputWasCalled)

    def test_adds_after_flush_are_not_buffered(self):
        self.objects.flush()
        self.objects.add('x')

        self.assertTrue(self.outputWasCalled)

    @property
    def outputWasCalled(self):
        return len(self.outputed) > 0
