import argparse, re, random, math, copy
from operator import attrgetter

POPULATION = 0
MUTATION = 0
GENERATIONS = 0
GROUP = 0

lookup_table = None

class Chromosome(object):
    """
    The chromosome is a representation
    of the genetic chromosome that is used
    to develop a schedule and calculate the
    makespan.

    The chromosome is represented as an array
    that is of size m * n, where m is the number
    of machines, and n is the number of jobs.

    Each operation is assigned a unique identifier
    with its cost. So Job 1 on Machine 1 would have
    an identifier of 1.

    The array represents the order in which operations
    are scheduled.

    A possible Chromosome might look like the following
    if there are 3 jobs and 3 machines

    [7, 5, 9, 3, 1, 8, 6, 2, 4]

    This means that schedule operation 7 first, then
    schedule operation 5, then schedule operation 9, and
    so on.
    """
    def __init__(self, sequence):
        self.sequence = sequence
        self._makespan = None

    @property
    def makespan(self):
        schedule = [[0 for j in lookup_table[0]] for i in lookup_table]

        for operation in self.sequence:
            job = id_to_job_index(operation)
            machine = id_to_machine_index(operation)
            op_cost = id_to_lookup(operation)

            column = [row[machine] for row in schedule]
            next_time = max(column)

            if max(schedule[job]) > next_time:
                next_time = max(schedule[job])

            schedule[job][machine]  = next_time + op_cost

        columns = zip(*schedule)
        makespan = 0
        for column in columns:
            cand = max(column)
            if cand > makespan:
                makespan = cand
        return makespan

class Population(object):
    def __init__(self, size):
        self.size = size
        self._members = []
        self._seed_population()

    def _seed_population(self):
        sequence_size = len(lookup_table) * len(lookup_table[0])
        for i in range(self.size):
            sequence = random.sample(range(1, sequence_size + 1), sequence_size)
            self._members.append(Chromosome(sequence))

    def evolve_population(self):
        (parent_one, parent_two) = self._selection()
        child = self._crossover(parent_one, parent_two)
        self._members.append(child)
        self.kill_weak()
        for member in self._members:
            if random.random() < MUTATION:
                self._mutate(member)
    
    def _crossover(self, parent_one, parent_two):
        (start_index, end_index) = random.sample(range(len(parent_one.sequence)), 2)
        
        child_seq = [None] * len(parent_one.sequence)
        
        for i in range(len(child_seq)):
            if start_index < end_index and i >= start_index and i <= end_index:
                child_seq[i] = parent_two.sequence[i]
            elif start_index > end_index:
                if not (i <= start_index and i >= end_index):
                    child_seq[i] = parent_one.sequence[i]

        for i in range(len(child_seq)):
            if parent_two.sequence[i] not in child_seq:
                for j in range(len(child_seq)):
                    if child_seq[j] is None:
                        child_seq[j] = parent_two.sequence[i]
                        break

        return Chromosome(child_seq)

    def _mutate(self, member):
        (index_one, index_two) = random.sample(range(len(member.sequence)), 2)

        member.sequence[index_one], member.sequence[index_two] = member.sequence[index_two], member.sequence[index_one]

    def _selection(self):
        num_to_select = math.floor(self.size * (GROUP/100))
        sample = random.sample(range(self.size), num_to_select)

        sample_members = sorted([self._members[i] for i in sample], key=attrgetter('makespan'))
        return sample_members[:2]

    def fittest(self, size):
        return sorted(self._members, key=attrgetter('makespan'))[:size]

    def kill_weak(self):
        weakest = max(self._members, key=attrgetter('makespan'))
        self._members.remove(weakest)

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
    arg_parser.add_argument('-gr', '--group', type=int, default=10,
                            help='The percentage of the group to use for crossover \
                                  selection. Default=10')
    return arg_parser.parse_args()

def parse_file(file_path):
    """
    Parse file returns a tuple of length two
    that contains the processing power of the machines
    """
    with open(file_path) as file:
        contents = file.read()
        machines = list(map(float, re.findall(r'MACHINE: ([0-9. ]*)', contents)[0].split(' ')))
        jobs = list(map(int, re.findall(r'JOBS: ([0-9 ]*)', contents)[0].split(' ')))

        return (machines, jobs)

def id_to_machine_index(id):
    transposed_id = id - 1
    return transposed_id % len(lookup_table)

def id_to_job_index(id):
    transposed_id = id - 1
    return transposed_id // len(lookup_table[0])

def id_to_lookup(id):
    """
    Looks up the value in the lookup table
    based on a unique identifier assigned to
    job and machine.

    For example, the Operation Job 1, Machine 1 would 
    have an id of 1. This would look that value up.
    """
    transposed_id = id - 1
    return lookup_table[transposed_id//len(lookup_table)][transposed_id%len(lookup_table[0])]

def create_lookup_table(machines, jobs):
    """
    Create a lookup table to lookup what each
    job will cost on a specific machine.

    The jobs are the rows, and the machines are the columns

    For example: To find the operation cost to perform
    job 1 on machine 1, you would access the data held
    in the lookup table at index [0][0]. To find the
    operation cost to run job 2 on machine 3, you would
    access the data held in the lookup table [1][2]

    Note: The index is of the machine and job is always
    one less then the ID of the job/machine
    """
    global lookup_table

    lookup_table = [[0 for i in jobs ] for i in machines]

    for i in range(0, len(machines)):
        for j in range(0, len(jobs)):
            lookup_table[j][i] = machines[i] * jobs[j]

def start_ga(population):
    best = 999999999
    for i in range(GENERATIONS):
        population.evolve_population()
        fittest = population.fittest(1)[0]
        if fittest.makespan < best:
            best = fittest.makespan 
        print(best)


if __name__ == '__main__':
    args = parse_args()
    POPULATION = args.population
    MUTATION = args.mutation
    GENERATIONS = args.generations
    GROUP = args.group

    machines, jobs = parse_file(args.file)
    create_lookup_table(machines, jobs)

    population = Population(POPULATION)
    start_ga(population)
