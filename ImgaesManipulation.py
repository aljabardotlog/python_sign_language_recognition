from PIL import Image

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

print("Training Running")
for i in range(0, 1000):
    resizeImage("Dataset/Training/A/a_" + str(i) + '.png')
    resizeImage("Dataset/Training/L/l_" + str(i) + '.png')
    resizeImage("Dataset/Training/Nama/nama_" + str(i) + '.png')
    resizeImage("Dataset/Training/Berapa/berapa_" + str(i) + '.png')
    resizeImage("Dataset/Training/Kamu/kamu_" + str(i) + '.png')
    resizeImage("Dataset/Training/Samasama/sama_" + str(i) + '.png')
    resizeImage("Dataset/Training/Saya/saya_" + str(i) + '.png')
    resizeImage("Dataset/Training/Sayang/sayang_" + str(i) + '.png')
    resizeImage("Dataset/Training/Terimakasih/terimakasih_" + str(i) + '.png')
    resizeImage("Dataset/Training/Umur/umur_" + str(i) + '.png')

print("Validation Running")
for i in range(0, 100):
    resizeImage("Dataset/Validation/A/a_" + str(i) + '.png')
    resizeImage("Dataset/Validation/L/l_" + str(i) + '.png')
    resizeImage("Dataset/Validation/Nama/nama_" + str(i) + '.png')
    resizeImage("Dataset/Validation/Berapa/berapa_" + str(i) + '.png')
    resizeImage("Dataset/Validation/Kamu/kamu_" + str(i) + '.png')
    resizeImage("Dataset/Validation/Samasama/sama_" + str(i) + '.png')
    resizeImage("Dataset/Validation/Saya/saya_" + str(i) + '.png')
    resizeImage("Dataset/Validation/Sayang/sayang_" + str(i) + '.png')
    resizeImage("Dataset/Validation/Terimakasih/terimakasih_" + str(i) + '.png')
    resizeImage("Dataset/Validation/Umur/umur_" + str(i) + '.png')


