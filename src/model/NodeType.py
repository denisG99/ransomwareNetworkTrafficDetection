"""""
 * This is an enumeration that decribe the node type.
 
 * The node types are: DECISION, LEAF, SPLIT or SUBSTITUTION.
"""""

from enum import Enum

class NodeType(Enum):
    DECISION = 0
    LEAF = 1
    SPLIT = 2
    SUBSTITUTION = 3