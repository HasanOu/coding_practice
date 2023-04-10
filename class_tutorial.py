"""This is based on this
# https://www.youtube.com/watch?v=ZDa-Z5JzLYM&list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc&ab_channel=CoreySchafer
"""

# =============================== create a class =======================================================================
# A class is a blueprint to create instances/objects.

class Employee:
    pass

emp_1 = Employee()   # we call this an instance of a class. Each instance is an object.
emp_2 = Employee()

print("print emp_1 and emp_2 instance")
print(emp_1)
print(emp_2)
print(emp_1 is emp_2)  # These are two different objects/instances from Empployee class

emp_1.first = 'Hasan'
emp_1.last = 'MN'

emp_2.first = 'Tannaz'
emp_2.last = 'FA'

# =================================================== create init method ===============================================

# init method is used to get attributes from an instance of a class
class Employee:
    def __init__(self, first, last, pay):
        # these are instance variables
        self.first = first
        self.last = last
        self.pay = pay
        self.email = self.first + self.last + '@company.com'

emp_1 = Employee('Hasan', 'MN', 10000)
emp_2 = Employee('Tannaz', 'FA', 5000)

print("print emp_1 and emp_2 first name using __init__ method")
print(emp_1.first)
print(emp_2.first)

# ================================= create a method inside a class =====================================================
# A method is simply a function within a class

class Employee:
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = self.first + self.last + '@company.com'

    def full_name(self):
        return f"{self.first}, {self.last}"

emp_1 = Employee('Hasan', 'MN', 10000)
emp_2 = Employee('Tannaz', 'FA', 5000)

print("print the output of full_name method")
print(Employee.full_name(emp_1))  # we can write it this way
print(emp_1.full_name())  # or we can write it this way (this is preferred way to do it)

# ================================== Instance variable versus Class variables ==========================================
# Class variables are shared across the class regardless of each instance
class Employee:
    raise_amount = 1.1       # This is a class variable
    def __init__(self, first, last, pay):
        self.first = first  # instance variable
        self.last = last
        self.pay = pay
        self.email = self.first + self.last + '@company.com'

    def apply_raise(self):
        # Both of these are correct. However, the second one is more flexible because we can change it for each instance
        # self.pay = int(self.pay*Employee.raise_amount)
        self.pay = int(self.pay*self.raise_amount)

emp_1 = Employee('Hasan', 'MN', 10000)
emp_2 = Employee('Tannaz', 'FA', 5000)

# For class variable, it initially checks if the instance has that attribute.
# Next, it will check if the class has that attribute

print("print __dict__ emp_1 instance and __dict__ for Employee class")
print(emp_1.__dict__)
print(Employee.__dict__)

# This changes class variables for all instances
print("print class variable and instance variable")
Employee.raise_amount = 1.5   # class attribute
print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)

# This only changes it for emp_1.
print("print class variable and instance variable")
emp_1.raise_amount = 1.3
print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)

# We use class variable in a situation where this does not depend on each instance
class Employee:
    raise_amount = 1.1
    num_of_emps = 0
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = self.first + self.last + '@company.com'

        # a class variable
        Employee.num_of_emps +=1

    def apply_raise(self):
        # self.pay = int(self.pay*Employee.raise_amount)
        self.pay = int(self.pay*self.raise_amount)

    def full_name(self):
        return f"{self.first}, {self.last}"

emp_1 = Employee('Hasan', 'MN', 10000)
emp_2 = Employee('Tannaz', 'FA', 5000)

print("print a class variable")
print(Employee.num_of_emps)

# ================================== Regular/instance methods, class methods, and static method ========================

# Regular/instance method: automatically takes instance as the argument (we use self)

# Class method:
# - automatically takes class as the argument (cls)
# - modifies the behaviour of the class using a decorator

# static method: static method does not pass anything autmatically

class Employee:
    raise_amount = 1.1
    num_of_emps = 0

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = self.first + self.last + '@company.com'

        # a class variable
        Employee.num_of_emps +=1

    def apply_raise(self):        # this is a regular/instance method
        # self.pay = int(self.pay*Employee.raise_amount)
        self.pay = int(self.pay*self.raise_amount)

    def full_name(self):
        return f"{self.first}, {self.last}"

    # Here we are working with a class instead of instance. It takes class and modifies
    @classmethod    # this is a class method
    def set_raise_amount(cls, amount):
        cls.raise_amount = amount

    @classmethod
    def from_string(cls, emp_str):  # this is a class method
        first, last, pay = emp_str.split("-")
        return cls(first, last, pay)

    # static method does not take self or cls. It is simply a function within the class
    @staticmethod
    def is_workday(day):
        if day.weekday == 5 or day.weekday == 6:
            return False
        else:
            return True

emp_1 = Employee('Hasan', 'MN', 10000)
emp_2 = Employee('Tannaz', 'FA', 5000)

emp_str_1 = 'John-Doe-70000'
emp_str_2 = 'Steve-Smith-30000'
emp_str_3 = 'Jane-Doe-90000'

print("print class method")
new_emp_1 = Employee.from_string(emp_str_1)
new_emp_2 = Employee.from_string(emp_str_2)
new_emp_3 = Employee.from_string(emp_str_3)
print(new_emp_1.first)

print("print static method")
import datetime
my_date = datetime.date(2017, 1, 5)
print(new_emp_1.is_workday(my_date))

# ========================= Inherritance ==========================
# Inheritance allows us to inherent attributes and methods from parent class
class Employee:
    raise_amount = 1.1
    num_of_emps = 0

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = self.first + self.last + '@company.com'

        # a class variable
        Employee.num_of_emps +=1

    def apply_raise(self):
        # self.pay = int(self.pay*Employee.raise_amount)
        self.pay = int(self.pay*self.raise_amount)

    def full_name(self):
        return f"{self.first}, {self.last}"

# inherent Employee attributes and methods from Employee class
class Developer(Employee):
    raise_amount = 1.5

    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay) # it takes first, last, and pay from the parent class
        # Employee.__init__(self, first, last, pay)
        self.prog_lang = prog_lang

class Manager(Employee):
    def __init__(self, first, last, pay, employee_list = None):
        super().__init__(first, last, pay)
        # Employee.__init__(self, first, last, pay)
        if employee_list is None:
            self.employee_list = []
        else:
            self.employee_list = employee_list

    def add_emp(self, emp):
        if emp not in self.employee_list:
            self.employee_list.append(emp)

    def remove_emp(self, emp):
        if emp in self.employee_list:
            self.employee_list.remove(emp)

    def full_name(self):
        return f"{self.first} + {self.last}"

dev_1 = Developer('Tannaz', 'FA', 10000, 'Java')
dev_2 = Developer('Hasan', 'MN', 50000, 'Python')

print(dev_1.first)

# will go into chain to find this information (dev_1.email does not exist in the dev class. It exists in the Employee)
print(dev_1.email)

print(help(Developer))  # provides a lot of good info
print(dev_1.prog_lang)

man_1 = Manager("Sue", "Smith", 90000, [dev_1])
man_1.add_emp(dev_2)
man_1.remove_emp(dev_1)

# Is instance tells if a object/instance is from a class
print(isinstance(man_1, Manager)) # '---> True'
print(isinstance(man_1, Employee))  # '---> True'
print(isinstance(man_1, Developer)) # '---> False'

# Is subclass tells if a class is a subclass of another class
print(issubclass(Manager, Employee)) # '---> True'
print(issubclass(Developer, Employee))  #'---> True'
print(issubclass(Manager, Developer)) # '---> False'

# ======================================= Special (Magic/Dunder) Methods ===============================================
class Employee:
    def __init__(self, first, last, pay):
        # these are instance variables
        self.first = first
        self.last = last
        self.pay = pay
        self.email = self.first + self.last + '@company.com'

    # unambiguous representation of the object used for the developer
    def __repr__(self):
        return f"Employee({self.first},{self.last},{self.pay})"

    # readable representation of the object used for the end user
    def __str__(self):
        return f"{self.first},{self.last}"

    # This will be used for + operator
    def __add__(self,other):
        return self.pay + other.pay

emp_1 = Employee('Hasan', 'MN', 10000)
emp_2 = Employee('Tannaz', 'FA', 5000)

# If __str__ is not deactivate, it only uses __str__ (not __repr__)
print(emp_1)

print("print repr and str")
print(repr(emp_1))
print(str(emp_1))
print(emp_1.__repr__())
print(emp_1.__str__())

print(emp_1+emp_2)

# ======================== Property Decorators: Getters, Setters, Deleters =============================================
class Employee:
    def __init__(self, first, last, pay):
        # these are instance variables
        self.first = first
        self.last = last
        self.pay = pay
        self.email = self.first + self.last + '@company.com'

    @property
    def email_address(self):
        return f"{self.first}{self.last}" + "email.com"

    @property
    def full_name(self):
        return f"{self.first} + {self.last}"

    @full_name.setter
    def full_name(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last

    @full_name.deleter
    def full_name(self):
        self.first = None
        self.last = None

emp_1 = Employee('Hasan', 'MN', 10000)
emp_2 = Employee('Tannaz', 'FA', 5000)
emp_1.first = 'Jim'

# This makes it run like an attribute instead of a method
# print(emp_1.email_address()) # you need to remove @property

print ("print property for email_address")
print(emp_1.email_address)
print(emp_1.first, emp_1.last)
print(emp_1.full_name)

print ("print setter for email_address")
emp_1.full_name = "Ebisa Wolega"
print(emp_1.first)
print(emp_1.last)
print(emp_1.email_address)

#
del emp_1.full_name
