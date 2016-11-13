import argparse, re

class Machine(object):
    def __init__(self, id, power_factor):
        self.id = id
        self.power_factor = power_factor

    def __repr__(self):
        return '{}: {}'.format(self.id, self.power_factor)

class Job(object):
    def __init__(self, id, time):
        self.id = id
        self.process_time = time

    def __repr__(self):
        return '{}: {}'.format(self.id, self.process_time)

class Instance(object):
    def __init__(self, machine, job):
        self.machine = machine
        self.job = job

class Schedule(object):
    def __init__(self):
        return

class Population(object):
    def __init__(self, machines, jobs, size):
        self.population_size = size

        self.members = self._seed_population(machines, jobs)

    def _seed_population(self, machines, jobs):
        print(machines)
        print(jobs)
        return

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--file',
                            help='Enter file that contains data')
    return arg_parser.parse_args()

def parse_file(file_path):
    """
    Parse file returns a tuple of length two
    that contains the processing power of the machines 
    """
    with open(file_path) as file:
        contents = file.read()
        machines = re.findall(r'MACHINE: ([0-9. ]*)', contents)[0].split(' ')
        jobs = re.findall(r'JOBS: ([0-9 ]*)', contents)[0].split(' ')

        machines = [Machine(index+1, float(i)) for index, i in enumerate(machines)]
        jobs = [Job(index+1, int(i)) for index, i in enumerate(jobs)]

        return {
            'machines': machines,
            'jobs': jobs
            }


if __name__ == '__main__':
    args = parse_args()
    data = parse_file(args.file)

    population = Population(data['machines'], data['jobs'], 100)
