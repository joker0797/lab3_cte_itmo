# LSB_DCT.py
from cv2 import cv2
import sys,numpy as np,itertools

class StegaException(Exception):
    pass

class LSB_DCT_steg():
    def __init__(self,img):
        self.image = img
        self.orirow,self.oricol,self.chan = img.shape
        # Добавляем некоторый пиксель, чтобы делить строку и столбец на 8
        if self.orirow % 8 != 0 or self.oricol %8 != 0:
            newRow = self.orirow+8-self.orirow%8
            newCol = self.oricol+8-self.oricol%8
            self.image = cv2.resize(self.image,(newCol,newRow))
        self.row,self.col,_ = self.image.shape
        # разбить изображение на 8 * 8 блоков
        self.blocks = self.break8()
        self.quant = np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])
    # разбить изображение на 8 * 8 блоков
    def break8(self):
        bImage,_,_ = cv2.split(self.image)
        bImage = np.float32(bImage)
        imgBlocks = [bImage[j:j+8, i:i+8] 
                    for (j,i) in itertools.product(range(0,self.row,8),range(0,self.col,8))]
        return imgBlocks

    # вернуть все группы, в которых сумма всей ширины = self.col
    def chunkRows(self,blocks,n):
        m = int(n)
        for i in range(0, len(blocks), m):
            yield blocks[i:i + m]

    def reshape(self,blocks):
        _,gImage,rImage = cv2.split(self.image)
        bImage = []
        for chunk in self.chunkRows(blocks,self.col/8):
            for numRow in range(8):
                for block in chunk:
                    bImage.extend(block[numRow])
        bImage = np.array(bImage).reshape(self.row,self.col)
        bImage = np.uint8(bImage)
        img = cv2.merge((bImage,gImage,rImage))
        return img
    
    # изменить значение в двоичную форму с фиксированным размером кусочка
    def binary_value(self,value,bitsize):
        binVal = bin(value)[2:]
        if len(binVal) > bitsize: 
            raise StegaException("Your message contain unexpected character!!!")
        while len(binVal) < bitsize:
            binVal = '0' + binVal
        return binVal

    def binary_text(self,mess):
        res = ""
        l = len(mess)
        res += self.binary_value(l,16)
        for ch in mess:
            c = ord(ch)
            res += self.binary_value(c,8)
        return res

    def embed(self,mess):
        # вычислить блоки коэффициентов DCT из изображения
        DCT_blocks = [np.round(cv2.dct(block)) for block in self.blocks]
        
        # разделить каждый коэффициент на величину квантования
        quantizedDCT = [np.round(DCT_block/self.quant) for DCT_block in DCT_blocks]
        # quantizedDCT = [np.round(DCT_block/1) for DCT_block in DCT_blocks]
        # перевести сообщение в двоичную форму
        mess = self.binary_text(mess)
        messIndex = 0
        # вставлять каждый бит сообщения в 1-й коэффициент каждого блока
        for block in quantizedDCT:
            DC = int(block[0,0])
            if DC % 2 == 0:
                DC ^= int(mess[messIndex])
            else:
                DC ^= int(mess[messIndex]) ^ 1
            block[0,0] = np.float32(DC)
            messIndex += 1
            if messIndex == len(mess): break
        if messIndex < len(mess)-1: raise StegaException("not enough spaces to embed")
        # умножить каждый коэффициент на значение квантования
        DCT_blocks = [block * self.quant for block in quantizedDCT]
        # DCT_blocks = [block * 1 for block in quantizedDCT]
        # рассчитать обратное DCT, чтобы получить значения изображения
        Img_blocks = [np.round(cv2.idct(block)) for block in DCT_blocks]
        # изменить форму блоков в форму изображения
        img = self.reshape(Img_blocks)
        # img = cv2.resize(img,(self.oricol,self.orirow))
        return img


    # прочитать все вставленные биты в блоках
    def read_bits(self,blocks):
        res = ""
        for block in blocks:
            c = int(block[0,0])
            if c % 2 == 0: res += '0'
            else: res += '1'
        return res

    def extract(self):
        # вычислить блоки коэффициентов DCT из изображения
        DCT_blocks = [np.round(cv2.dct(block)) for block in self.blocks]
        # разделить каждый коэффициент на величину квантования
        quantizedDCT = [np.round(DCT_block/self.quant) for DCT_block in DCT_blocks]
        # quantizedDCT = [np.round(DCT_block/1) for DCT_block in DCT_blocks]
        
        length = self.read_bits(quantizedDCT[:16])
        length = int(length,2)
        cursor = 16
        mess = ""
        for _ in range(length):
            c = self.read_bits(quantizedDCT[cursor:cursor+8])
            c = int(c,2)
            ch = chr(c)
            mess += ch
            cursor += 8
            if cursor >= len(quantizedDCT): 
                raise StegaException("Your image hasn't been embed!!")
        return mess
        
    def statistic(self):
        # вычислить блоки коэффициентов DCT из изображения
        DCT_blocks = [np.round(cv2.dct(block)) for block in self.blocks]
        # разделить каждый коэффициент на величину квантования
        quantizedDCT = [np.round(DCT_block/self.quant) for DCT_block in DCT_blocks]
        import matplotlib.pyplot as plt
        histogram = []
        for block in quantizedDCT:
            for i in range(8):
                for j in range(8):
                    val = float(block[i,j])
                    histogram.append(val)
        plt.hist(histogram,bins = 100)
        plt.show()