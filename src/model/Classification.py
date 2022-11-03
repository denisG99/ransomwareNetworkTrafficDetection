"""""
 * This is an enumeration that decribe the node type.

 * The node types are: BENIGNWARE, MALWARE, NONE.
"""""

from enum import Enum

class Classification(Enum):
    BENIGNWARE = 0
    MALWARE = 1
    NONE = 2