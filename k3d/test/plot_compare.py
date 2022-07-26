import os
import pytest
import sys
from PIL import Image
from base64 import b64encode
from io import BytesIO
from pixelmatch.contrib.PIL import pixelmatch


def prepare():
    while len(pytest.plot.objects) > 0:
        pytest.plot -= pytest.plot.objects[-1]

    pytest.plot.clipping_planes = []
    pytest.plot.colorbar_object_id = 0
    pytest.plot.grid_visible = True
    pytest.plot.camera_mode = 'trackball'
    pytest.plot.camera = [2, -3, 0.2,
                          0.0, 0.0, 0.0,
                          0, 0, 1]
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.camera_reset()


def compare(name, only_canvas=True, threshold=0.2, camera_factor=1.0):
    pytest.headless.sync(hold_until_refreshed=True)

    if camera_factor is not None:
        pytest.headless.camera_reset(camera_factor)

    screenshot = pytest.headless.get_screenshot(only_canvas)

    result = Image.open(BytesIO(screenshot))
    img_diff = Image.new("RGBA", result.size)
    reference = None

    if os.path.isfile('./test/references/' + name + '.png'):
        reference = Image.open('./test/references/' + name + '.png')
    else:
        if sys.platform == 'win32':
            if os.path.isfile('./test/references/win32/' + name + '.png'):
                reference = Image.open('./test/references/win32/' + name + '.png')
        else:
            if os.path.isfile('./test/references/linux/' + name + '.png'):
                reference = Image.open('./test/references/linux/' + name + '.png')

    if reference is None:
        reference = Image.new("RGBA", result.size)

    mismatch = pixelmatch(result, reference, img_diff, threshold=threshold, includeAA=True)

    if mismatch >= threshold:
        with open('./test/results/' + name + '.k3d', 'wb') as f:
            f.write(pytest.plot.get_binary_snapshot(1))
        result.save('./test/results/' + name + '.png')
        reference.save('./test/results/' + name + '_reference.png')
        img_diff.save('./test/results/' + name + '_diff.png')

        print(name, mismatch, threshold)

        if 'CI' in os.environ and os.environ['CI'] == 'true':
            print(name)
            print(b64encode(open('./test/results/' + name + '.png', 'rb').read()))

    assert mismatch < threshold
