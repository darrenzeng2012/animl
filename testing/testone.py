import graphviz
from gen_samples import *

#st = viz_boston()
st = viz_iris(fancy=False)
g = graphviz.Source(st)
g.view()
