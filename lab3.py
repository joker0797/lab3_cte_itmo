#lab3.py
from LSB_DCT import *
from analysis import *
import argparse

#config аргументы командной строки
parse = argparse.ArgumentParser(description= "lab3 staganography- LTD N3349")
parse.add_argument('-i',metavar='ifile',type=str,dest= "in_text",help= "name of input text file")
parse.add_argument('-ii',metavar='inimg',type=str,dest= "in_img",help= "name of input image file",required= True)
parse.add_argument('-o',metavar='ofile',type=str,dest= "out_file",help= "name of output image/text file")
parse.add_argument('option',metavar='opts',type=str,help= "name of option (embed/extract/PSNR/detection)")
args = parse.parse_args()

# вставлять текстовые данные в изображение методом LSB-DCT
if args.option == "embed":
    # читать текст для вставки
    text = open(args.in_text,"r",encoding="UTF-8").read()
    # получить данные и запустить LSB-DCT stegano
    data_in = cv2.imread(args.in_img)
    steg = LSB_DCT_steg(data_in)
    data_out = steg.embed(text)
    # записать данные изображения в файл
    cv2.imwrite(args.out_file,data_out)

# извлекать текстовые данные из изображения методом LSB-DCT
elif args.option == "extract":
    # получить данные и запустить инвертированный LSB-DCT стегано
    data_in = cv2.imread(args.in_img)
    steg = LSB_DCT_steg(data_in)
    text = steg.extract()
    # записать извлеченный текст в файл
    open(args.out_file,"w",encoding="UTF-8").write(text)

# рассчитать PSNR 2 изображения
elif args.option == "PSNR":
    # получить 2 входных изображения и запустить PSNR
    data1 = cv2.imread(args.in_img)
    data2 = cv2.imread(args.out_file)
    analize = Analysis(data1,data2)
    PSNR = analize.PSNR()
    # показать PSNR
    print("PSNR of your 2 images is:",end = " ")
    print(PSNR)

# нарисовать PSNR-изображение с изображением, вставленным символами 1,2, ....
elif args.option == "graph":
    # читать вводимый текст
    text = open(args.in_text,"r",encoding="UTF-8").read()
    # получить данные входных изображений
    data_in = cv2.imread(args.in_img)
    # data_out = cv2.imread(args.out_file)
    in_text = ""
    PSNRs = []
    numWords = []
    for i in range(len(text)):
        in_text += text[i]
        print(in_text)
        steg = LSB_DCT_steg(data_in) # метод init LSB-DCT
        data_out = steg.embed(in_text)
        data_in = cv2.imread(args.in_img)
        analize = Analysis(data_in,data_out)
        # сохранить выходные данные в массив
        numWords.append(i)
        PSNR = analize.PSNR()
        PSNRs.append(PSNR)
    # рисовать PSNR графику
    import matplotlib.pyplot as plt
    plt.plot(numWords,PSNRs)
    plt.xlabel("Number of words hided in image")
    plt.ylabel("PSNR value in Db")
    plt.title("Graphic PSNR with Number of words ratio")
    plt.show()
# запустить простую атаку на метод LSB-DCT
elif args.option == "detection":
    # получить данные входных изображений
    data_in = cv2.imread(args.in_img)
    steg = LSB_DCT_steg(data_in)
    steg.statistic()