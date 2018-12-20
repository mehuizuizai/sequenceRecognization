from PIL import Image
import  os
xs = 28
ys =28
# RANGE_DIR = [0,1,2,3,4,5,6,7,8,9,"A","C","D","E","F","K","L","M","R","S","U","V","X","Y"]
RANGE_DIR = ["T"]
def resize(num):
    # strfile ="E:/胎号/338-2_new/分割后/" + str(num)
    # outfile = "E:/胎号/338-2_new/分割后resize/" + str(num)
    strfile = "E:/胎号所有数字_字母/总/" + str(num)
    outfile = "E:/胎号所有数字_字母/总_new/"  + str(num)
    if os.path.exists(strfile):
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        for rt, dirs, files in os.walk('E:/胎号所有数字_字母/总/'+str(num)+'/'):
            for file in files:
                im = Image.open('E:/胎号所有数字_字母/总/'+str(num)+'/'+file)
                (x,y) = im.size  # read image size
                out_file = im.resize((xs,ys),Image.ANTIALIAS)
                out_file.save("E:/胎号所有数字_字母/总_new/" + str(num)+'/' +file)


if __name__ == '__main__':
    for i in RANGE_DIR:
     resize(i)