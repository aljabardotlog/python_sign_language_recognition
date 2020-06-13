f = open("text1.txt", "r")
for x in f:
    a = x.replace("\n",'0')
    if a!='0':
        print(a)