'''
Redundant file used to practise using classes. Doesn't do anything of note.
'''

import numpy as np
import matplotlib.pyplot as plt

class body1():

    # Whatever you give the class body1, it goes to __init__ as "identifier".
    # This is then put into the box called "self", and anywhere you want to
    # grab identifier again, you have to call it via self.name
    def __init__(self, identifier):
        self.name = identifier
        self.moons= []

    # Method that grabs the name of the planet and prints a statement
    def namecalling(self):
        print(str(self.name)+" is a planet")
    
    # Method that grabs the moon list within the "self box" and appends a name
    # given to it when the method is called
    def add_moon(self, moon_name):
        self.moons.append(moon_name)

body_1 = body1("Jupiter")
body_1.namecalling()
body_1.add_moon("Io")
print(body_1.moons[0])