class Employee:

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.raise_amount = 1.04

    def full_name(self):
        return f"{self.first} {self.last}"

    def apply_raise(self):
        self.pay = self.pay*self.raise_amount

    @classmethod
    def set_raise_mount(cls, amount):
        pass

emp_1 = Employee('Hasan', 'Manzour', 50000)
emp_1.raise_amount = 2

emp_2 = Employee('Tannaz', 'Fatemieh', 40000)
emp_3 = Employee('Donald', 'Obama', 40000)

print(emp_1.apply_raise())
print(emp_1.pay)
