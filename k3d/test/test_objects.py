import unittest
from ..objects import SingleObject, _Attribute

NoneType = type(None)


class MockObject(SingleObject):
    foo = _Attribute((str, NoneType), str, 'foo')


class TestMockObject(unittest.TestCase):
    def setUp(self):
        self.obj = MockObject(foo=None)

    def test_iteration_returns_object(self):
        for obj in self.obj:
            self.assertTrue(isinstance(obj, MockObject))

    def test_can_add_another_object(self):
        self.obj += MockObject(foo=None)

        for obj in self.obj:
            self.assertTrue(isinstance(obj, MockObject))

    def test_can_add_many_objects(self):
        obj = MockObject(foo=None)
        self.obj += obj + obj

        for obj in self.obj:
            self.assertTrue(isinstance(obj, MockObject))

    def test_keeps_non_empty_values(self):
        obj = MockObject(foo='bar')
        self.assertTrue('foo' in obj.data)

    def test_hides_empty_values(self):
        obj = MockObject(foo=None)
        self.assertTrue('foo' not in obj.data)
