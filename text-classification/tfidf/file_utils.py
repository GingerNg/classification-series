import pickle


def readFile(path):
    with open(path, 'r', errors='ignore',encoding="gbk") as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        return content

def readFile2(path):
    with open(path, 'r', errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        return content

def saveFile(path, result):
    result = result
    with open(path, 'w', errors='ignore') as file:
        file.write(result)


def readBunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
        #pickle.load(file)
        #函数的功能：将file中的对象序列化读出。
    return bunch

def writeBunch(path, bunchFile):
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)


def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList