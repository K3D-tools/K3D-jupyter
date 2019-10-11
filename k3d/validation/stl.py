import re

from traitlets import Bytes, Unicode, TraitError

float_re = r'''
(?:
    [-+]? # optional sign
    (?:
         (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc.
         |
         (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc.
    )
    # followed by optional exponent part if desired
    (?: [Ee] [+-]? \d+ ) ?
)
'''

stl_re = r'''
    solid .* \n  # header
    (?:
        \s* facet \s normal (?: \s ''' + float_re + ''' ){3}
        \s* outer \s loop
        (?:
            \s* vertex (?: \s ''' + float_re + ''' ){3}
        ){3}
        \s* endloop
        \s* endfacet
    ) + # at least 1 facet.
    \s* endsolid
    \s* $ # e.g. trailing newline
'''

ascii_stl = re.compile(stl_re, re.VERBOSE)


class AsciiStlData(Unicode):
    def validate(self, owner, stl):
        stl = super(AsciiStlData, self).validate(owner, stl)

        if ascii_stl.match(stl) is None:
            raise TraitError('Given string is not valid ASCII STL data.')

        return stl


class BinaryStlData(Bytes):
    HEADER = 80
    FACET_SIZE = 50

    def validate(self, owner, stl):
        stl = super(BinaryStlData, self).validate(owner, stl)

        if (len(stl) - self.HEADER) % self.FACET_SIZE != 0:
            raise TraitError(
                'Given bytestring has wrong length ({}) for Binary STL data.'
                .format(len(stl))
            )

        return stl
