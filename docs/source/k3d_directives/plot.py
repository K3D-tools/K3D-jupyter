import importlib.util
import os
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective


class K3D_Plot(SphinxDirective):
    """ Sphinx class for execute_code directive
    """
    has_content = False
    required_arguments = 0
    optional_arguments = 2

    option_spec = {
        'filename': directives.path,
        'screenshot': directives.flag
    }

    def run(self):
        """ Executes python code for an RST document, taking input from content or from a filename
        :return:
        """

        filename = self.options.get('filename')
        path = self.env.doc2path(self.env.docname)
        code_path = os.path.join(os.path.dirname(path), filename)
        image_filepath = os.path.join(os.path.dirname(path),
                                      os.path.splitext(os.path.basename(filename))[0] + '.png')

        if 'screenshot' in self.options:
            if os.path.isfile(image_filepath):
                return []

        spec = importlib.util.spec_from_file_location("module.name", code_path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        code_results = foo.generate()

        output = []

        if 'screenshot' in self.options:
            with open(image_filepath, 'wb') as image:
                image.write(code_results)
        else:
            output.append(nodes.raw('', code_results, format='html'))

        return output
