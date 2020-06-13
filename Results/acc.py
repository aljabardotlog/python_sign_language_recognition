f = open("text1.txt", "r")
no=1
for x in f:
    txt = x.split(" | ")
    if txt[0] == '| Adam':
        ll = txt[2]
        xx = ll.replace(": ","|")
        yy = xx.split('--')
        pp = yy[0]
        zz = pp.replace(" - ","|")
        rr = zz.split('|')
        print(str(no)+","+rr[3])
        no+=1