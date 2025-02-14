import os
import shutil as shu

image_path = 'picture-gene'
image_dest = 'picture-gene-all'
c = 0
for sub in os.listdir(image_path):
    for img in os.listdir(image_path+'/'+sub):
        dest = img[:-4]+sub[-4:]+'.png'
        shu.copy(image_path+'/'+sub+'/'+img, image_dest+'/'+dest)
        c += 1
        if c%100==0:
            print(c)

for i in os.listdir(image_dest):
    if '_gtsub1' in i:
        os.rename(image_dest+'/'+i, image_dest+'/'+i[:-8]+'.png')
for i in os.listdir(image_dest):
    if 'gtsub' in i:
        os.remove(image_dest+'/'+i)
     