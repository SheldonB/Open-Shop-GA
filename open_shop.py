import argparse, re

class Machine(object):
    def __init__(self, power_factor):
        self.power_factor = power_factor


class Job(object):
    def __init__(self, time):
        self.process_time = time

class Instance(object):
    def __init__(self, machines, jobs):
        self.machines = machines
        self.jobs = jobs

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--file',
                            help='Enter file that contains data')
    return arg_parser.parse_args()

def parse_file(file_path):
    with open(file_path) as file:
        contents = file.read()
        machines = re.findall(r'MACHINE: ([0-9. ]*)', contents)[0].split(' ')
        jobs = re.findall(r'JOBS: ([0-9 ]*)', contents)[0].split(' ')
        print(machines)
        print(jobs)

if __name__ == '__main__':
    args = parse_args()
    parse_file(args.file)
