import os
files = []

for dirName, subdirList, fileList in os.walk('test'):
    dirName = str(dirName)

    for f in fileList:
        f = dirName + "\\" + f
        files.append(f)
for f in files:
    try:
        print(f)
    except:
        pass
