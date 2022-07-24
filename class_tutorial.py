# What is object-oriented programming?
# Object-oriented programming (OOP) is a computer programming model that organizes software design around data, or
# objects, rather than functions and logic.
# An object can be defined as a data field that has unique attributes and behavior.

# Classes are user-defined data types that act as the blueprint for individual objects, attributes and methods.
# Objects are instances of a class created with specifically defined data. Objects can correspond to real-world objects
# or an abstract entity.
# Methods are functions that are defined inside a class that describe the behaviors of an object. Each method contained.
# Attributes are defined in the class template and represent the state of an object.

# Encapsulation. This principle states that all important information is contained inside an object and only select
# information is exposed. The implementation and state of each object are privately held inside a defined class. Other
# objects do not have access to this class or the authority to make changes. They are only able to call a list of public
# functions or methods. This characteristic of data hiding provides greater program security and avoids unintended data
# corruption.

# Abstraction. Objects only reveal internal mechanisms that are relevant for the use of other objects, hiding any
# unnecessary implementation code. The derived class can have its functionality extended. This concept can help
# developers more easily make additional changes or additions over time.

# Inheritance. Classes can reuse code from other classes. Relationships and subclasses between objects can be assigned,
# enabling developers to reuse common logic while still maintaining a unique hierarchy. This property of OOP forces a
# more thorough data analysis, reduces development time and ensures a higher level of accuracy.

# Polymorphism. Objects are designed to share behaviors and they can take on more than one form. The program will
# determine which meaning or usage is necessary for each execution of that object from a parent class,
# reducing the need to duplicate code. A child class is then created, which extends the functionality of the parent
# class. Polymorphism allows different types of objects to pass through the same interface.

# position, name, age, level, salary

se1_list = ["Hasan", 23, "Senior", "500"]
se2_list = ["Tannaz", 25, "Senior", "400"]


# inherits, extends, and override
class Employee:
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary
        self._promotion = None # protected
        self.__fire = None # private

    def work(self):
        return f"{self.name} is an Employee with age {self.age} ans salary {self.salary} is working"

    # getters and setters
    def get_promotion(self):
        return self._promotion

    def set_promotion(self, value):
        if value <= 10000:
            value = 10000
        else:
            value = 20000
        self._promotion = value


class Designer(Employee):
    def __init__(self, name, age, level, salary):
        super().__init__(name, age, salary)
        self.level = level

    def work(self):
        return f"{self.name} is designer with age {self.age} ans salary {self.salary} and level {self.level} is working"


class SoftwareEngineer(Employee):
    raise_amount = 1.04

    def __init__(self, name, age, salary):
        super().__init__(name, age, salary)

    def employee_info(self):
        return f"This programmer: name={self.name}, age={self.age}, salary={self.salary}"

    def empl_language(self, language):
        return f"{language}"

    # deunder method agic methods are not meant to be invoked directly by you, but the invocation happens internally
    # from the class on a certain action.
    def __str__(self):
        information = f"This programmer: name={self.name}, age={self.age}, salary={self.salary}"
        return information

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

    def __add__(self, other):
        return self.age - other.age

    @staticmethod
    def entry_salary(age):
        if age <= 25:
            return 200000
        else:
            return 500000


# instance
se1 = SoftwareEngineer("Hasan", 35, 500000)
print(SoftwareEngineer.raise_amount)
print(se1.raise_amount)
print(se1.employee_info())
print(se1.empl_language('python'))

se2 = Designer("Hasan", 40, "Senior", 500000)
se3 = Designer("Hasan", 35, "Senior", 500000)
print(se1 == se2 == se3)

# override "+" deunder method
print(se1 + se2)
print(se1.entry_salary(35))

print(se1.work())



# recap
# create class (blueprint)
# create instance (object)
# class vs instance
# instance attribute: defined in __init__(self)
# class attribute
# ----------
# instance method(self)
# can take arguments and return values
# special "deunder" (__add__(self) __str__(self), __eq__(self))
# Inheritance: ChildClass (BaseClass)
# inherit, extend, overtide
# super().__init__()
# polymorphism
# The word polymorphism means having many forms. In programming, polymorphism means the same function
# name (but different signatures) being used for different types.
# -----------
# Encapsulation
# private attribute
# private function
# getter and setter
# -----------
# Abstraction
# Abstraction in python is that feature in OOP concept wherein the user is kept unaware of the basic implementation of
# a function property. The user is only able to view basic functionalities whereas the interbal detauls are hidden.
# ----------
# Properties