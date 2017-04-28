# How to

## Add loader strategy for new object type

1. Define a loader strategy function in `app/providers/<PROVIDER>/objects/<TYPE>.js`

    ### Parameters

    **config**  
    An instance of `K3D.Config` containing an object configuration passed to loader

    ### Return value

    A `Promise` which should eventually resolve with the value of object ready to render

2. Define a interaction function in `app/providers/<PROVIDER>/interactions/<TYPE>.js`

    > **Note**  
    This step is required for interactive objects only

    ### Parameters

    **arg1, arg2, ...**  
    An arbitrary list of arguments

    ### Return value

    An object with `onHover(intersect, viewMode)` and `onClick(intersect, viewMode)` methods.

    > **Note**  
    This function should be explicitly called by loader strategy function and it's return value should be assigned to result object's `interactions` property.

## Change 3D library (provider) used

After reimplementing what's required in a separate folder inside `app/providers/<PROVIDER>` you will be able to use new provider.
Additional changes in `app/index.html` might be required to make the "build" and "serve" tasks working.

