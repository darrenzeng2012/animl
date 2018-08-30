import graphviz
from gen_samples import *

st = viz_boston()
g = graphviz.Source(st)
g.view()
