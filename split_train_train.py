import  os
import shutil
RANGE_DIR= ("B","G","H","I","J","P","Z")

for i in RANGE_DIR:
    dir = 'E:/胎号所有数字_字母/shengxia/%s/' % i
    targe_dir_train ='E:/胎号所有数字_字母/shengxia_train/%s/' % i
    targe_dir_test ='E:/胎号所有数字_字母/shengxia_test/%s/' % i
    a = os.listdir(dir)
    if not os.path.exists(targe_dir_train):
            os.makedirs(targe_dir_train)
    if not os.path.exists(targe_dir_test):
            os.makedirs(targe_dir_test)
    for j in a[:int(len(a)*8//10)]:
            shutil.copyfile(dir+j,targe_dir_train+j)
    for k in a[int(len(a)*8//10)+1:]:
            shutil.copyfile(dir+k,targe_dir_test+k)

