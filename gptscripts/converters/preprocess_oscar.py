import os
import json 

TYPE = r"train"
FILE_PATH = r'/data/nlp-zh/oscarcorpus/' + TYPE
SAVE_PATH = r'/data/converted'

def write_line(text):
    with open(os.path.join(SAVE_PATH, TYPE + ".json"), "a", encoding='utf-8') as f:
        f.write(json.dumps({'text':text}, ensure_ascii=False) + '\n')

def main():
    files = os.listdir(FILE_PATH)
    for file in files:
        with open(os.path.join(FILE_PATH, file),encoding='utf-8') as f:
            lines = f.readlines()
            temp = []
            for line in lines:
                if line == "\n":
                    write_line("".join(temp))
                    temp = []
                else:
                    temp.append(line.strip('\n').strip())
                    
if __name__ == "__main__":
    main()
    