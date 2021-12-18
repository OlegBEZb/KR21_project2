from typing import List, Tuple, Dict
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.readwrite import XMLBIFReader
import math
import itertools
import pandas as pd
from copy import deepcopy
from functools import reduce


class BayesNet:

    def __init__(self) -> None:
        # initialize graph structure
        self.structure = nx.DiGraph()

    # LOADING FUNCTIONS ------------------------------------------------------------------------------------------------
    def create_bn(self, variables: List[str], edges: List[Tuple[str, str]], cpts: Dict[str, pd.DataFrame]) -> None:
        """
        Creates the BN according to the python objects passed in.
        
        :param variables: List of names of the variables.
        :param edges: List of the directed edges.
        :param cpts: Dictionary of conditional probability tables.
        """
        # add nodes
        [self.add_var(v, cpt=cpts[v]) for v in variables]

        # add edges
        [self.add_edge(e) for e in edges]

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            raise Exception('The provided graph is not acyclic.')

    def load_from_bifxml(self, file_path: str) -> None:
        """
        Load a BayesNet from a file in BIFXML file format. See description of BIFXML here:
        http://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/

        :param file_path: Path to the BIFXML file.
        """
        # Read and parse the bifxml file
        with open(file_path) as f:
            bn_file = f.read()
        bif_reader = XMLBIFReader(string=bn_file)

        # load cpts
        cpts = {}
        # iterating through vars
        for key, values in bif_reader.get_values().items():
            values = values.transpose().flatten()
            n_vars = int(math.log2(len(values)))
            worlds = [list(i) for i in itertools.product([False, True], repeat=n_vars)]
            # create empty array
            cpt = []
            # iterating through worlds within a variable
            for i in range(len(values)):
                # add the probability to each possible world
                worlds[i].append(values[i])
                cpt.append(worlds[i])

            # determine column names
            columns = bif_reader.get_parents()[key]
            columns.reverse()
            columns.append(key)
            columns.append('p')
            cpts[key] = pd.DataFrame(cpt, columns=columns)
        
        # load vars
        variables = bif_reader.get_variables()
        
        # load edges
        edges = bif_reader.get_edges()

        self.create_bn(variables, edges, cpts)

    # METHODS THAT MIGHT ME USEFUL -------------------------------------------------------------------------------------

    def get_children(self, variable: str) -> List[str]:
        """
        Returns the children of the variable in the graph.
        :param variable: Variable to get the children from
        :return: List of children
        """
        return [c for c in self.structure.successors(variable)]

    def get_cpt(self, variable: str) -> pd.DataFrame:
        """
        Returns the conditional probability table of a variable in the BN.
        :param variable: Variable of which the CPT should be returned.
        :return: Conditional probability table of 'variable' as a pandas DataFrame.
        """
        try:
            return self.structure.nodes[variable]['cpt']
        except KeyError:
            raise Exception('Variable not in the BN')

    def get_all_variables(self) -> List[str]:
        """
        Returns a list of all variables in the structure.
        :return: list of all variables.
        """
        return [n for n in self.structure.nodes]

    def get_all_cpts(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of all cps in the network indexed by the variable they belong to.
        :return: Dictionary of all CPTs
        """
        cpts = {}
        for var in self.get_all_variables():
            cpts[var] = self.get_cpt(var)

        return cpts

    @property
    def structure_unidirected(self):
        return self.structure.to_undirected()

    def find_observed_ancestors(self, observed):
        """
        Traverse the graph, find all nodes that have observed descendants.
        Args:
            observed: a set of strings, names of the observed nodes.
        Returns:
            a set of strings for the nodes' names for all nodes
            with observed descendants.
        """
        nodes_to_visit = set(observed)  ## nodes to visit
        observed_ancestors = set()  ## observed nodes and their ancestors

        ## repeatedly visit the nodes' parents
        while len(nodes_to_visit) > 0:
            next_node = nodes_to_visit.pop()
            if next_node not in observed_ancestors:
                nodes_to_visit = nodes_to_visit | set(pred for pred in self.structure.predecessors(next_node))
            observed_ancestors = observed_ancestors | set([next_node])

        return observed_ancestors

    # TODO: extract reachable
    # TODO: wrap to make each parameter a set
    def d_separated(self, start, end, observed):
        """
        Check whether start and end are d-separated given observed.
        This algorithm mainly follows the "Reachable" procedure in
        Koller and Friedman (2009),
        "Probabilistic Graphical Models: Principles and Techniques", page 75.
        Args:
            start: a string, name of the first query node
            end: a string, name of the second query node
            observed: a list of strings, names of the observed nodes.
        """

        ## all nodes having observed descendants.
        obs_anc = self.find_observed_ancestors(observed)

        ## Try all active paths starting from the node "start".
        ## If any of the paths reaches the node "end",
        ## then "start" and "end" are *not* d-separated.
        ## In order to deal with v-structures,
        ## we need to keep track of the direction of traversal:
        ## "up" if traveled from child to parent, and "down" otherwise.
        nodes_to_visit = [(start, "up")]
        visited = set()  ## keep track of visited nodes to avoid cyclic paths
        reachable_from_start = set()

        while len(nodes_to_visit) > 0:
            (nname, direction) = nodes_to_visit.pop()
            print('considering node:', (nname, direction))

            ## skip visited nodes
            if (nname, direction) not in visited:
                # ## if reaches the node "end", then it is not d-separated
                # if nname not in observed and nname == end:
                #     return False
                if nname not in observed:
                    reachable_from_start = reachable_from_start | set([nname])

                visited.add((nname, direction))

                ## if traversing from children, then it won't be a v-structure
                ## the path is active as long as the current node is unobserved
                if direction == "up" and nname not in observed:
                    for parent in self.structure.predecessors(nname):
                        nodes_to_visit.append((parent, "up"))
                    for child in self.structure.successors(nname):
                        nodes_to_visit.append((child, "down"))

                ## if traversing from parents, then need to check v-structure
                elif direction == "down":
                    ## path to children is always active
                    if nname not in observed:
                        for child in self.structure.successors(nname):
                            nodes_to_visit.append((child, "down"))
                    ## path to parent forms a v-structure
                    if nname in obs_anc:
                        for parent in self.structure.predecessors(nname):
                            nodes_to_visit.append((parent, "up"))
        if end in reachable_from_start:
            return True
        else:
            return False

    def get_interaction_graph(self):
        """
        Returns a networkx.Graph as interaction graph of the current BN.
        :return: The interaction graph based on the factors of the current BN.
        """
        # Create the graph and add all variables
        int_graph = nx.Graph()
        [int_graph.add_node(var) for var in self.get_all_variables()]

        # connect all variables with an edge which are mentioned in a CPT together
        for var in self.get_all_variables():
            involved_vars = list(self.get_cpt(var).columns)[:-1]
            for i in range(len(involved_vars)-1):
                for j in range(i+1, len(involved_vars)):
                    if not int_graph.has_edge(involved_vars[i], involved_vars[j]):
                        int_graph.add_edge(involved_vars[i], involved_vars[j])
        return int_graph

    @staticmethod
    def get_compatible_instantiations_table(instantiation: pd.Series, cpt: pd.DataFrame):
        """
        Get all the entries of a CPT which are compatible with the instantiation.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series(("A", True), ("B", False))
        :param cpt: cpt to be filtered
        :return: table with compatible instantiations and their probability value
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        compat_indices = cpt[var_names] == instantiation[var_names].values
        compat_indices = [all(x[1]) for x in compat_indices.iterrows()]
        compat_instances = cpt.loc[compat_indices]
        return compat_instances

    def update_cpt(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Replace the conditional probability table of a variable.
        :param variable: Variable to be modified
        :param cpt: new CPT
        """
        self.structure.nodes[variable]["cpt"] = cpt

    @staticmethod
    def reduce_factor(instantiation: pd.Series, cpt: pd.DataFrame) -> pd.DataFrame:
        """
        Creates and returns a new factor in which all probabilities which are incompatible with the instantiation
        passed to the method to 0.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True}, {"B": False})
        :param cpt: cpt to be reduced
        :return: cpt with their original probability value and zero probability for incompatible instantiations
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        if len(var_names) > 0:  # only reduce the factor if the evidence appears in it
            new_cpt = deepcopy(cpt)
            incompat_indices = cpt[var_names] != instantiation[var_names].values
            incompat_indices = [any(x[1]) for x in incompat_indices.iterrows()]  # check any non-negative values
            new_cpt.loc[incompat_indices, 'p'] = 0.0
            return new_cpt
        else:
            return cpt

    @staticmethod
    def sum_out(factor, variables):
        Y = [X for X in factor.columns if X not in variables and X != 'p']
        return factor.groupby(Y)['p'].sum().to_frame().reset_index()

    @staticmethod
    def multiply_factors(f1, f2):
        on = [c for c in f1.columns if c in f2.columns and c != 'p']
        result = pd.merge(f1, f2, on=on, how='outer')
        result['p'] = result['p_x'] * result['p_y']
        return result.drop(['p_x', 'p_y'], axis=1)

    @staticmethod
    def VE_PR1(bn, Q):
        S = bn.get_all_cpts()
        for var in [v for v in bn.get_all_variables() if v not in Q]:
            print('processing', var)
            f_k = {k: df for k, df in S.items() if var in df}
            [display(df) for k, df in f_k.items()]
            f = reduce(bn.multiply_factors, list(f_k.values()))
            print('f')
            display(f)
            f_i = bn.sum_out(f, set([var]))
            print('f_i')
            display(f_i)
            for k in f_k:
                del S[k]
            S[var + '*'] = f_i
            print('S\n', S)
        return reduce(bn.multiply_factors, list(S.values()))

    def draw_structure(self) -> None:
        """
        Visualize structure of the BN.
        """
        nx.draw(self.structure, with_labels=True, node_size=3000)
        plt.show()

    # BASIC HOUSEKEEPING METHODS ---------------------------------------------------------------------------------------

    def add_var(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Add a variable to the BN.
        :param variable: variable to be added.
        :param cpt: conditional probability table of the variable.
        """
        if variable in self.structure.nodes:
            raise Exception('Variable already exists.')
        else:
            self.structure.add_node(variable, cpt=cpt)

    def add_edge(self, edge: Tuple[str, str]) -> None:
        """
        Add a directed edge to the BN.
        :param edge: Tuple of the directed edge to be added (e.g. ('A', 'B')).
        :raises Exception: If added edge introduces a cycle in the structure.
        """
        if edge in self.structure.edges:
            raise Exception('Edge already exists.')
        else:
            self.structure.add_edge(edge[0], edge[1])

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            self.structure.remove_edge(edge[0], edge[1])
            raise ValueError('Edge would make graph cyclic.')

    def del_var(self, variable: str) -> None:
        """
        Delete a variable from the BN.
        :param variable: Variable to be deleted.
        """
        self.structure.remove_node(variable)

    def del_edge(self, edge: Tuple[str, str]) -> None:
        """
        Delete an edge form the structure of the BN.
        :param edge: Edge to be deleted (e.g. ('A', 'B')).
        """
        self.structure.remove_edge(edge[0], edge[1])
