import marshal

# Example data to be dumped
data1 = {"name": "John", "age": 30}
data2 = {"name": "Alice", "age": 25}
data3 = {"name": "Bob", "age": 35}

# File path to write Marshal data
file_path = "../output.marshal"

# Open the file in write binary mode
with open(file_path, "wb") as marshal_file:
    # Dump the first set of data into the file
    marshal.dump(data1, marshal_file)

    # Dump the second set of data into the file
    marshal.dump(data2, marshal_file)

    # Dump the third set of data into the file
    marshal.dump(data3, marshal_file)

import marshal

# File path to read Marshal data
file_path = "../output.marshal"

# Open the file in read binary mode
with open(file_path, "rb") as marshal_file:
    # Read the first set of data from the file
    data1 = marshal.load(marshal_file)

    # Read the second set of data from the file
    data2 = marshal.load(marshal_file)

    # Read the third set of data from the file
    data3 = marshal.load(marshal_file)

    data4 = marshal.load(marshal_file)

print(data1)
print(data2)
print(data3)
print(data4)
