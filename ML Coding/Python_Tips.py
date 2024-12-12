################################################################
                        Default Parameter Settings
################################################################
class Player:
    def __init__(self, name, items=[])
            self.name = name
            self.items = items
            # print(id(self.items))

p1 = Player('Alice')
p2 = Player('Bob')
p3 = Player('Charles', ['sword'])

p1.items.append('armor')
p2.items.append('sword')

print(p1.items)
# Returns ['armor', 'sword']


----- Correct way -------
class Player:
    def __init__(self, name, items=None)
            self.name = name
            if items is None:
                self.items = []
            else:
                self.items = items
            # print(id(self.items))

'''
Tip 2: When dealing with None, the best practice is to use the is operator for comparisons with None
'''
if a:
  print('Not None')
# This checks whether a is truthy. In Python, None, 0, False, empty collections (e.g., [], {}), and empty strings ("") evaluate to False.
# Drawback: This approach doesn’t specifically test for None. It might give misleading results if a is something else that evaluates to False (e.g., 0 or []).

if a == None:
  print('None')
# This works because None is equal to itself.
# Drawback: It’s less idiomatic and might be prone to issues if custom objects override the == operator in a way that makes them equal to None.
