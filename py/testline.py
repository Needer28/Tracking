from PIL import Image


img = Image.new("RGB",(5,5))###创建一个5*5的图片
pixTuple = (255,0,255,15)###三个参数依次为R,G,B,A   R：红 G:绿 B:蓝 A:透明度
for i in range(5):
    for j in range(5):
        img.putpixel((i,j),pixTuple)
img.save("bb.png")
