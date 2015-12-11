# How To

## Add support for new object type

> **Note**  
New object type has to be already supported by [k3d](https://github.com/K3D-tools/K3D)
(see: [How to add loader strategy for new object type](https://github.com/K3D-tools/K3D/blob/master/HOW-TO.md))

1. Add new object class in `k3d/objects.py`

    ### Requirements

    The new class has to:
    * have a name matching an object type in k3d
    * extend `SingleObject`
    * define a list of attributes, each being an instance of `_Attribute` class

    ### _Attribute class

    #### Signature

        def __init__(self, expected_type, cast, path)

    **expected_type**  
    An expected attribute value type or list of types, e.g. `str` or `(int, float)`  
    Used with `isinstance` to validate when attribute value is being set.

    **cast**  
    A function to call for a value, when object is being serialized to JSON to be passed to *k3d*.

    **path**  
    A string representing attribute location within the JSON document passed to k3d, e.g. `rootLevel` or `nested/path`.

    #### Transform

    `_Attribute` class provides `transform` method allowing converting a value of unsupported input type into supported one.  
    May be called multiple times allowing chaining transformations.

        def transform(self, input_type, transform)

    **input_type**  
    A type to which transformation should be applied.

    **transform**  
    A function to call for a value which should return supported type (or another transformable type).

2. Add new factory method in `k3d/factory.py`

    This new method:
    * should have a name matching an object type in k3d using *lower_case_with_underscores* notation
    * can accept arbitrary arguments
    * has to return an instance of a class created in #1
