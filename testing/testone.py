import graphviz
from gen_samples import *

#st = viz_boston()
st = viz_iris()
g = graphviz.Source(st)
g.view()
