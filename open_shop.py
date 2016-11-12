import argparse, re

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--file',
                            help='Enter file that contains data')
    return arg_parser.parse_args()

def parse_file(file_path):
    with open(file_path) as file:
        contents = file.readline()
        print(contents)
        machines = re.match(r'([0-9]*[.][0-9]*)', contents)

if __name__ == '__main__':
    args = parse_args()
    parse_file(args.file)
