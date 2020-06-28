### classes.py
### this file defines the object types for the simulation


#-------------------------------------------------------------------------

class Customer:
    """
    A class used to represent a single customer node.

    Initialization Parameters:
    ----------
    ID : identifying label (str)
    x : x-coordinate of the customer location (float)
    y : y-coordinate of the customer location (float)
    d : customer's realized demand

    Other Attributes
    ----------
    dsplit : a dictionary of (truck ID, demand filled) pairs to track split demand
    loc : the (x,y) location of the customer (tuple)
    type : the node class, i.e. customer

    """

    def __init__(self, x, y, ID, d):
        self.ID = str(ID)
        self.x = float(x)
        self.y = float(y)
        self.d = float(d)
        self.dsplit = {}
        self.loc = (self.x, self.y)
        self.type = 'customer'


#-------------------------------------------------------------------------

class Depot:
    """
    A class used to represent a single depot node.

    Initialization Parameters
    ----------
    ID : identifying label (str)
    x : x-coordinate of the customer location (float)
    y : y-coordinate of the customer location (float)

    Other Attributes
    ----------
    loc : the (x,y) location of the customer (tuple)
    type : the node class, i.e. depot

    """

    def __init__(self, ID, x, y):
        self.ID = str(ID)
        self.x = float(x)
        self.y = float(y)
        self.loc = (self.x, self.y)
        self.type = 'depot'


#-------------------------------------------------------------------------

class Route:
    """
    A class used to represent a delivery route

    Initialization Parameters:
    ----------
    sequence : dictionary of position and customer node objects in greater network
    ID : identifying label (str)

    Other Attributes
    ----------
    n : number of customers visited on route
    d : total demand filled for route customers (should be adjusted if customer demands are split)
    customers : ordered list of customer objects in route
    type : the node class, i.e. route

    """

    def __init__(self, sequence, ID):
        self.ID = ID
        self.d = sum([cust.d for cust in list(sequence.values())])
        self.sequence = sequence
        self.customers = list(sequence.values())
        self.n = len(sequence)
        self.type = 'route'

