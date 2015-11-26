import unittest
from ..objects import SingleObject


class MockObject(SingleObject):
    def __init__(self, data=None):
        if data is None:
            data = {}

        super(MockObject, self).__init__(data)


class TestMockObject(unittest.TestCase):
    def setUp(self):
        self.obj = MockObject()

    def test_iteration_returns_object(self):
        for obj in self.obj:
            self.assertTrue(isinstance(obj, MockObject))

    def test_can_add_another_object(self):
        self.obj += MockObject()

        for obj in self.obj:
            self.assertTrue(isinstance(obj, MockObject))

    def test_can_add_many_objects(self):
        obj = MockObject()
        self.obj += obj + obj

        for obj in self.obj:
            self.assertTrue(isinstance(obj, MockObject))

    def test_keeps_non_empty_values(self):
        obj = MockObject({'foo': 'bar'})
        self.assertTrue('foo' in obj.__dict__)

    def test_hides_empty_values(self):
        obj = MockObject({'foo': None})
        self.assertTrue('foo' not in obj.__dict__)
