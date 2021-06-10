import mtg_jamendo_dataset.scripts.commons as commons
from zipfile import ZipFile
import os
import sys

def getInstrumentFilePaths(directory, split): 
    allPaths = [] #list of file path for each dir
    foundPaths = [] #the instruments paths we are looking for

    for d in os.listdir(directory):
        path = os.path.join(directory, d)

        if os.path.isdir(path):
            paths = []

            for p in os.listdir(path):
                fpath = os.path.join(path, p)
                if os.path.isfile(fpath):
                    paths.append(fpath)

            allPaths.append(paths)

    if(split == "train"):
        labels = 'mtg_jamendo_dataset/data/splits/split-0/autotagging_instrument-train.tsv'
    elif(split == "validation"):
        labels = 'mtg_jamendo_dataset/data/splits/split-0/autotagging_instrument-validation.tsv'
    elif(split == "test"):
        labels = 'mtg_jamendo_dataset/data/splits/split-0/autotagging_instrument-test.tsv'

    tracks, tags, extra = commons.read_file(labels) 

    print("searching for file paths...")
    for track in tracks:
        for f in allPaths[int(str(track)[-2:])]:
            if int(f.split('/')[-1].split('.')[0]) == track:
                foundPaths.append(f)
                break
                
    return foundPaths        
  
def main():
    if len(sys.argv) != 3:
        return
  
    paths = getInstrumentFilePaths(sys.argv[-1], sys.argv[1])
  
    print("# of tracks: " + str(len(paths)))
  
    for i in range(15):
        print("Writing zip #" + str(i) + " out of 15")
        with ZipFile(sys.argv[-1] + sys.argv[1] + str(i) + 'SetMTG.zip','w') as zip:
            if(i == 14):
                for file in paths[(i*1000):]:
                    zip.write(file)
            else:
                for file in paths[(i*1000):((i*1000) + 1000)]:
                    zip.write(file)
   
if __name__ == "__main__":
    main()
