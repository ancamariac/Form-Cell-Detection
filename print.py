test_list = [[1,0,0], [0, 2], [1], [3,4,5,6], [1,0,1,1], [2], [5]]

for iterator, mini_list in enumerate(test_list):
    print("=== Question " + str(iterator) + " ===")
    for option, element in enumerate(mini_list):
        print("option " + chr(option + 65) + ": " + str(element))
    print("\n")
    