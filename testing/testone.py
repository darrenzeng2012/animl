import graphviz
from gen_samples import *

#st = viz_boston(fancy=False, orientation='TD')
#st = viz_iris(fancy=True, orientation='TD')
st = viz_digits(fancy=True, orientation='LR')
g = graphviz.Source(st)
g.view()
