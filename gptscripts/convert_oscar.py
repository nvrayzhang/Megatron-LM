import os
import json 
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",type=str)
    parser.add_argument("--output_path",type=str)
    args = parser.parse_args()
    return args

def write_line(text, args):
    with open(os.path.join(args.output_path, "pl_oscar_zh.json"), "a", encoding='utf-8') as f:
        f.write(json.dumps({'text':text}, ensure_ascii=False) + '\n')

def main():
    args = get_args()
    for curDir, dirs, files in os.walk(str(args.input_path), topdown=False):
        for file in files: 
            with open(os.path.join(curDir, file), encoding='utf-8') as f:
                lines = f.readlines()
                temp = []
                for line in lines:
                    if line == "\n":
                        write_line("".join(temp), args)
                        temp = []
                    else:
                        temp.append(line.strip('\n').strip())
                    
if __name__ == "__main__":
    main()
    