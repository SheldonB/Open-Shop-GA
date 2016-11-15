import argparse, re, copy, random

from operator import attrgetter

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

    def __repr__(self):
        return '(Machine: {}, Job:{})'.format(self.machine.id, self.job.id)

class Schedule(object):
    def __init__(self, matrix):
        self.matrix = matrix
        self._makespan = None
    
    @property
    def makespan(self):
        if self._makespan is None:
            for i in range(len(self.matrix[0])):
                for j in range(len(self.matrix)):
                    machine = self.matrix[j][i].machine
                    job = self.matrix[j][i].job

                    if job.finish_time > machine.total_time:
                        machine.total_time = job.finish_time + (machine.power_factor * job.process_time)
                    else:
                        machine.total_time = machine.total_time + (machine.power_factor * job.process_time)

                    job.finish_time = machine.total_time

            self._makespan = max([row[0].machine.total_time for row in self.matrix])
        return self._makespan


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
            for row in s_copy:
                row[rand_col_1], row[rand_col_2] = row[rand_col_2], row[rand_col_1]
            members.append(Schedule(s_copy))
        
        return members

    def _crossover(parent_one, parent_two):
        """
        Crossover method for the genetic algorithm.
        The method takes two members of the population
        and returns a child member of the population.
        """
        raise NotImplementedError()

    def _mutate(member):
        """
        Mutation method for the genetic algorithm.
        The method takes a member of the population 
        that is chosen based on the mutation rate.

        That member is then mutated, and a new member
        is returned
        """
        raise NotImplementedError()

    def evolve_population():
        """
        Evolve population will run generation of the
        genetic algorithm.
        """
        raise NotImplementedError()

    def fittest(self, size):
        return sorted(self.members, key=attrgetter('makespan'))

    def kill_weak(self):
        weakest = max(self.members, key=attrgetter('makespan'))
        self.members.remove(weakest)

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--file',
                            help='Enter file that contains data')
    arg_parser.add_argument('-p', '--population', type=int, default=100,
                            help='Population size of the genetic algorithm. Default=100')
    arg_parser.add_argument('-m', '--mutation', type=float, default=2.5,
                            help='Mutation rate of the genetic algorithm. Default=2.5')
    arg_parser.add_argument('-g', '--generations', type=int, default=100,
                            help='The number of generations for the GA to run. Default=100')
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

def start_ga(population):
    for i in range(GENERATIONS):
        print('WOOO GA')

if __name__ == '__main__':
    args = parse_args()
    POPULATION = args.population
    MUTATION = args.mutation
    GENERATIONS = args.generations
    data = parse_file(args.file)

    population = Population(data['machines'], data['jobs'], POPULATION)
    start_ga(population)

