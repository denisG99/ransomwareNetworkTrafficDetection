"""""
 * This is an enumeration that decribe the node type.
 
 * The node types are: DECISION, LEAF or SPLIT.
"""""

from enum import Enum

class NodeType(Enum):
    DECISION = 0
    LEAF = 1
    SPLIT = 2