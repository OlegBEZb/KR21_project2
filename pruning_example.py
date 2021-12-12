def pruning(self, query , evidence):
 """
 Given a set of query variables Q and evidence E, function node-prunes the
 Bayesian network
 :return: returns pruned Bayesian network
 """
var = self.bn.get_all_variables()
 # node pruning
for v in var:
   if (v not in query) and (v not in evidence) and (not self.bn.get_children(v)):
       self.bn.del_var(v)
 # TODO: add that the CPT must be updated.
 # edge pruning
for e in evidence: 
     children = self.bn.get_children(e)
for c in children:
     self.bn.del_edge([e, c])
 # TODO: update CPT