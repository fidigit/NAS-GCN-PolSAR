import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import k_means_

def gas(mean, std, x):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x - mean)**2 / std**2)#计算正太分布的似然函数

def mrf(path):
    img = np.array(plt.imread(path))
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgGray = imgGray / 255
    imgCopy = imgGray.copy()
    imgpixel = (imgCopy.flatten()).reshape((imgGray.shape[0]*imgGray.shape[1], 1))
    kind = 3
    kmeans = k_means_.KMeans(n_clusters=kind)#先使用kmeans完成第一次初始划分
    label = kmeans.fit(imgpixel)
    imgLabel = np.array(label.labels_).reshape(imgGray.shape)
    plt.figure()
    plt.imshow(imgLabel, cmap="gray")
    plt.show()
    imgMrf = np.zeros_like(imgLabel)
    cycle = 2#迭代次数
    c = 0
    sumList = [0] * kind
    numList = [0] * kind
    MeanList = [0] * kind
    stdSumList = [0] * kind
    stdList = [0] * kind
    for i in range(1, imgLabel.shape[0] - 1):
        for j in range(1, imgLabel.shape[1] - 1):
            x = imgLabel[i, j]
            sumList[x] += imgGray[i, j]
            numList[x] += 1
    for k in range(kind):
        MeanList[k] = sumList[k] / numList[k]#计算u（均值）

    for i in range(1, imgLabel.shape[0] - 1):
        for j in range(1, imgLabel.shape[1] - 1):
            x = imgLabel[i, j]
            stdSumList[x] += (imgGray[i, j] - MeanList[x]) ** 2#计算西格玛（方差）
    for i in range(kind):
        stdList[i] = np.sqrt(stdSumList[i] / numList[i])

    while c < cycle:
        for i in range(1, imgLabel.shape[0] - 1):
            for j in range(1, imgLabel.shape[1] - 1):
                uList = [0] * kind
                for k in range(kind):
                    template = np.ones((3, 3)) * k#绘制9宫格的内存空间
                    template[1, 1] = np.inf#保证所有数据能够显示，而不是用省略号表示，np.inf表示一个足够大的数
                    u = np.exp(- np.sum(template == imgLabel[i - 1: i + 2, j - 1: j + 2]) + 8)
                    gas(MeanList[k], stdList[k], imgGray[i, j])#计算z
                    #上面这整个代码完成了u这个像素点的后验概率的计算
                    uList[k] = u
                    sumList[k] += imgGray[i, j]
                    numList[k] += 1
                imgMrf[i, j] = uList.index(max(uList))
        for i in range(kind):
            MeanList[i] = sumList[i] / numList[i]#重新计算每类的均值
        for i in range(1, imgLabel.shape[0] - 1):
            for j in range(1, imgLabel.shape[1] - 1):
                x = imgLabel[i, j]
                stdSumList[x] += (imgGray[i, j] - MeanList[x]) ** 2#再次计算方差
        for i in range(kind):
            stdList[i] = np.sqrt(stdSumList[i] / numList[i])
        imgLabel = imgMrf.copy()
        c += 1
        print("第{}代结束".format(c))

    SavePath = "./result.png"
    img = cv2.normalize(1 - imgLabel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(SavePath,img)

    return SavePath

if __name__ == '__main__':
    mrf("1.png")