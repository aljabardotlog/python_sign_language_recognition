f = open("text.txt", "r")
for x in f:
    text = x.split('')
    print(text[0])
