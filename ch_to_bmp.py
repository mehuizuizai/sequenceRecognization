from PIL import  Image
import  shutil
import  cv2
import  os
# os.mkdir('./0_new_bmp/')

def ch_to_bmp(num):
    outfile =  "./number3/" + str(num)+'/'
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    for rt, dirs, files in os.walk( "./number2/" + str(num)+'/'):
        for file in files:
            file_split = file.split('.')
            print(file)
            lena = Image.open( "./number2/" + str(num)+'/'+file)
            lena_1 = lena.convert("L")
            lena_1.save(os.path.join( "./number3/" + str(num)+'/', os.path.basename(file_split[0]+'.bmp')))
if __name__ == '__main__':
    for i in range(0,10):
        ch_to_bmp(i)