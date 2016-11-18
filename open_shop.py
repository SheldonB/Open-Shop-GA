import argparse, re, copy, random, math

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
            job_time = [0 for i in range(len(self.matrix[0]))]
            machine_time = [0 for i in range(len(self.matrix))]

            for i in range(len(self.matrix[0])):
                for j in range(len(self.matrix)):
                    machine = self.matrix[j][i].machine
                    job = self.matrix[j][i].job

                    if job_time[job.id - 1] > machine_time[machine.id - 1]:
                        machine_time[machine.id - 1] = job_time[job.id - 1] + \
                                (machine.power_factor * job.process_time)
                    else:
                        machine_time[machine.id - 1] = machine_time[machine.id - 1] + \
                                (machine.power_factor * job.process_time)

                    job_time[job.id - 1] = machine_time[machine.id - 1]

            self._makespan = max(machine_time)

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
            (rand_col_1, rand_col_2) = random.sample(range(0, len(jobs) - 1) , 2)
            for row in s_copy:
                row[rand_col_1], row[rand_col_2] = row[rand_col_2], row[rand_col_1]
            members.append(Schedule(s_copy))

        return members

    def _crossover(self, parent_one, parent_two):
        """
        Crossover method for the genetic algorithm.
        The method takes two members of the population
        and returns a child member of the population.
        """
        # Initialze the child matrix
        child_matrix = [[ 0 for j in range(len(parent_one.matrix[0]))]
                                for i in range(len(parent_one.matrix))]

        # Choose random indicies for crossover
        (rand_col_1, rand_col_2) = random.sample(range(0, len(parent_one.matrix[0])) , 2)

        # If random column 1 is bigger then
        # random column 2, then swap them
        # just to make things easier
        if rand_col_1 > rand_col_2:
            rand_col_1, rand_col_2 = rand_col_2, rand_col_1

        # Fill the columns of the child with the columns
        # of parent one between the two indicies
        for i in range(rand_col_1, rand_col_2):
            for j, row in enumerate(parent_one.matrix):
                child_matrix[j][i] = row[i]

        # Transpose the matrix of parent two, so we
        # now have an array of all columns, that can
        # be compared to the columns
        parent_two_columns = list(map(list, zip(*parent_two.matrix)))

        # I am not a fan of this, but it converts the child
        # matrix to a reduced column matrix of just job numbers
        child_matrix_columns = []
        for column in zip(*child_matrix):
            if column[0] == 0:
                child_matrix_columns.append(list(column))
            else:
                child_matrix_columns.append([i.job.id for i in column])

        for cand_column in parent_two_columns:
            r_cand_column = [i.job.id for i in cand_column]

            if r_cand_column not in child_matrix_columns:
                for i in range(len(child_matrix[0])):
                    if child_matrix[0][i] == 0:
                        for j in range(len(child_matrix)):
                            child_matrix[j][i] = cand_column[j]
                        break

        return Schedule(child_matrix)

    def _mutate(self, member):
        """
        Mutation method for the genetic algorithm.
        The method takes a member of the population
        that is chosen based on the mutation rate.

        That member is then mutated, and a new member
        is returned
        """
        (index_one, index_two) = random.sample(range(0, len(member.matrix)), 2)
        for row in member.matrix:
            row[index_one], row[index_two] = row[index_two], row[index_one]


        # temp_id_1 = member.matrix[index_one][0].machine.id
        # temp_id_2 = member.matrix[index_two][0].machine.id
        # member.matrix[index_one], member.matrix[index_two] = member.matrix[index_two], \
                                                             # member.matrix[index_one]

        # for i in range(len(member.matrix)):
            # member.matrix[index_one][i].machine.id = temp_id_1
            # member.matrix[index_two][i].machine.id = temp_id_2
        member._makespan = None

    def evolve_population(self):
        """
        Evolve population will run generation of the
        genetic algorithm.
        """
        (parent_one, parent_two) = self._selection()

        child = self._crossover(parent_one, parent_two)

        self.members.append(child)

        for member in self.members:
            if random.random() < MUTATION:
                self._mutate(member)
        self.kill_weak()
        fit = self.fittest(1)[0]
        for row in fit.matrix:
            print([i.job.id for i in row])
        print(fit.makespan)
        # fit = sorted(self.members, key=attrgetter('makespan'))[0]

        # weak = sorted(self.members, key=attrgetter('makespan'))[-1]
        # print(fit.makespan, weak.makespan)

    def _selection(self):
        num_to_select = math.floor(len(self.members) * (GROUP/100))
        sample = random.sample(range(0, len(self.members)), num_to_select)

        sample_members = sorted([self.members[i] for i in sample], key=attrgetter('makespan'))
        return sample_members[:2]

    def fittest(self, size):
        return sorted(self.members, key=attrgetter('makespan'))[:size]

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
        population.evolve_population()

MUTATION = 0
GENERATIONS = 0
GROUP = 0

if __name__ == '__main__':
    args = parse_args()
    POPULATION = args.population
    MUTATION = args.mutation
    GENERATIONS = args.generations
    GROUP = args.group
    data = parse_file(args.file)

    population = Population(data['machines'], data['jobs'], POPULATION)
    start_ga(population)

