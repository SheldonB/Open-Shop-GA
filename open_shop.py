import argparse, re, copy, random

class Machine(object):
    def __init__(self, id, power_factor):
        self.id = id
        self.power_factor = power_factor
        self.total_time = 0

    def __repr__(self):
        return '{}: {}'.format(self.id, self.power_factor)

class Job(object):
    def __init__(self, id, time):
        self.id = id
        self.process_time = time
        self.finish_time = 0

    def __repr__(self):
        return '{}: {}'.format(self.id, self.process_time)

class Instance(object):
    def __init__(self, machine, job):
        self.machine = machine
        self.job = job

    def __repr__(self):
        return '(Machine: {}, Job:{})'.format(self.machine.id, self.job.id)

class Schedule(object):
    def __init__(self, matrix):
        # self.makespan = None
        self.matrix = matrix
    
    @property
    def makespan(self):
        for i in range(len(self.matrix[0])):
            for j in range(len(self.matrix)):
                machine = self.matrix[j][i].machine
                job = self.matrix[j][i].job

                if job.finish_time > machine.total_time:
                    machine.total_time = job.finish_time + (machine.power_factor * job.process_time)
                else:
                    machine.total_time = machine.total_time + (machine.power_factor * job.process_time)

                job.finish_time = machine.total_time

        return max([row[0].machine.total_time for row in self.matrix])

class Population(object):
    def __init__(self, machines, jobs, size):
        self.population_size = size

        self.members = self._seed_population(machines, jobs)

    def _seed_population(self, machines, jobs):
        members = []
        # Generate a Latin Square Matrix. That has no repeating
        # Columns and rows.
        schedule = []
        for i in range(0, len(machines)):
            row = [Instance(machines[i], job) for job in jobs]
            for j in range(0, i):
                to_rotate = row.pop(0)
                row.append(to_rotate)
            schedule.append(row)
        members.append(Schedule(schedule))  

        # To generate random members for the population
        # We can just swap the columns of the population
        for i in range(1, self.population_size):
            s_copy = copy.deepcopy(schedule)
            rand_col_1 = random.randint(0, len(jobs) - 1)
            rand_col_2 = random.randint(0, len(jobs) - 1)
            print(rand_col_1, rand_col_2)
            for row in s_copy:
                row[rand_col_1], row[rand_col_2] = row[rand_col_2], row[rand_col_1]
            members.append(Schedule(s_copy))
        
        return members

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

    for member in population.members:
        print(member.makespan)
