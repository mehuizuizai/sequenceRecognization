import os
import shutil

sdir = 'E:/胎号/338_new_cut/'#来源
ddir = 'E:/胎号/338_2_分好类/'#目的

path = os.listdir(sdir)

for i in path:
    sourceFile = sdir+i
    targetDir = ddir+i[i.find('_')+1]+'/'
    targetFile = targetDir+i
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    shutil.copyfile(sourceFile, targetFile)