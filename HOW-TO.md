# How To

## Add support for new object type

> **Note**  
New object type has to be already supported by the [K3D JS provider](js/src/providers/threejs/objects)  
(see: [How to add loader strategy for new object type](js/HOW-TO.md)).

1. Add a new object class in `k3d/objects.py`

    ### Requirements

    The new class has to:
    * have a name matching an object type in k3d
    * have a read-only `type` class variable, of type `traitlets.Unicode` with the default value
      set to the above name
    * extend `k3d.objects.Drawable`
    * define a list of class variables, each derived from the `traitlets.TraitType` class

2. Add a new helper function in `k3d/k3d.py`

    This new function should:
    * have a name matching the object type in k3d using *lower_case_with_underscores* notation
    * accept arbitrary arguments
    * return an instance of the class created in #1
