import numpy as np
import pandas as pd
import graphviz
from numpy.distutils.system_info import f2py_info
from sklearn import tree
from sklearn.datasets import load_boston, load_iris
from matplotlib.figure import figaspect
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
from animl.trees import *
from numbers import Number
import matplotlib.patches as patches

YELLOW = "#fefecd" # "#fbfbd0" # "#FBFEB0"
BLUE = "#D9E6F5"
GREEN = "#cfe2d4"
DARKBLUE = '#313695'
DARKGREEN = '#006400'
LIGHTORANGE = '#fee090'
LIGHTBLUE = '#a6bddb'
GREY = '#444443'

#dark_colors = [DARKBLUE, DARKGREEN, '#a50026', '#fdae61', '#c51b7d', '#fee090']

color_blind_friendly_colors = [
    None, # 0 classes
    None, # 1 class
    ["#fefecd","#D9E6F5"], # 2 classes
    ["#fefecd","#D9E6F5",'#a1dab4'], # 3 classes
    ["#fefecd","#D9E6F5",'#a1dab4','#fee090'], # 4
    ["#fefecd","#D9E6F5",'#a1dab4','#41b6c4','#fee090'], # 5
    ["#fefecd",'#c7e9b4','#41b6c4','#2c7fb8','#fee090','#f46d43'], # 6
    ["#fefecd",'#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#fdae61'], # 7
    ["#fefecd",'#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#fdae61'], # 8
    ["#fefecd",'#ece7f2','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#045a8d','#fdae61'], # 9
    ["#fefecd",'#e0f3f8','#313695','#fee090','#4575b4','#fdae61','#abd9e9','#74add1','#f46d43','#d73027'] # 10
]

max_class_colors = len(color_blind_friendly_colors)-1

def dtreeviz(tree_model, X_train, y_train, feature_names, target_name, class_names=None,
             precision=1, orientation="TD", show_edge_labels=False, fancy=True):
    def round(v,ndigits=precision):
        return format(v, '.' + str(ndigits) + 'f')

    def node_name(node : ShadowDecTreeNode) -> str:
        if node.feature_name() is None:
            return f"node{node.id}"
        node_name = ''.join(c for c in node.feature_name() if c not in string.punctuation)+str(node.id)
        node_name = re.sub("["+string.punctuation+string.whitespace+"]", '_', node_name)
        return node_name

    def split_node(name, node_name, split):
        if fancy:
            html = f"""<table border="0">
            <tr>
                    <td port="img" fixedsize="true" width="202.5" height="90"><img src="/tmp/node{node.id}.svg"/></td>
            </tr>
            </table>"""
        else:
            html = f"""<font face="Helvetica" color="#444443" point-size="12">{name}@{split}</font>"""
        gr_node = f'{node_name} [margin="0" shape=none label=<{html}>]'
        return gr_node


    def regr_leaf_node(node):
        value = node.prediction()
        if fancy:
            html = f"""<table border="0" CELLPADDING="0" CELLBORDER="0" CELLSPACING="0">
            <tr>
                    <td colspan="3" port="img" fixedsize="true" width="87" height="90"><img src="/tmp/node{node.id}.svg"/></td>
            </tr>
            </table>"""
            # this code makes big bubble around leaf:
            if False:
                margin = prop_size(node.nsamples(),
                                   counts=shadow_tree.leaf_sample_counts())
                return f'leaf{node.id} [margin="{margin}" style=filled fillcolor="white" shape=circle label=<{html}>]'
            return f'leaf{node.id} [margin="0" shape=plain label=<{html}>]'
        else:
            margin = prop_size(node.nsamples(),
                               counts=shadow_tree.leaf_sample_counts())
            html = f"""<font face="Helvetica" color="#444443" point-size="11">{target_name}<br/>{round(value)}</font>"""
            return f'leaf{node.id} [margin="{margin}" style=filled fillcolor="{YELLOW}" shape=circle label=<{html}>]'

    def class_leaf_node(node):
        counts = node.class_counts()
        predicted_class = np.argmax(counts)
        predicted = predicted_class
        if class_names is not None:
            predicted = class_names[predicted_class]
        ratios = counts / node.nsamples()  # convert counts to ratios totalling 1.0
        ratios = [round(r, 3) for r in ratios]
        color_spec = ["{c};{r}".format(c=color_values[i], r=r) for i, r in
                      enumerate(ratios)]
        color_spec = ':'.join(color_spec)
        if n_classes > max_class_colors:
            color_spec = YELLOW
        html = f"""<font face="Helvetica" color="black" point-size="12">{predicted}<br/>&nbsp;</font>"""
        margin = prop_size(node.nsamples(), counts = shadow_tree.leaf_sample_counts())
        style = 'wedged' if n_classes <= max_class_colors else 'filled'
        return f'leaf{node.id} [height=0 width="0.4" margin="{margin}" style={style} fillcolor="{color_spec}" shape=circle label=<{html}>]'

    def class_legend_html():
        elements = []
        for i,cl in enumerate(class_values):
            html = f"""
            <tr>
                <td border="0" cellspacing="0" cellpadding="0"><img src="/tmp/legend{i}.svg"/></td>
                <td align="left"><font face="Helvetica" color="{GREY}" point-size="12">{class_names[cl]}</font></td>
            </tr>
            """
            elements.append(html)
        return f"""
        <table border="0" cellspacing="0" cellpadding="0">
        <tr>
            <td border="0" colspan="2"><font face="Helvetica" color="{GREY}" point-size="12"><b>{target_name}</b></font></td>
        </tr>
        {''.join(elements)}
        </table>
        """

    ranksep = ".22"
    if orientation=="TD":
        ranksep = ".4"


    shadow_tree = ShadowDecTree(tree_model, X_train, y_train,
                                feature_names=feature_names, class_names=class_names)

    n_classes = shadow_tree.nclasses()
    color_values = color_blind_friendly_colors[n_classes]

    # Fix the mapping from target value to color for entire tree
    class_values = shadow_tree.unique_target_values
    colors = {v:color_values[i] for i,v in enumerate(class_values)}

    figsize = (4.5, 2)

    y_range = (min(y_train)*1.03, max(y_train)*1.03) # same y axis for all

    if shadow_tree.isclassifier():
        draw_legend_boxes(shadow_tree, "/tmp/legend")

    internal = []
    for node in shadow_tree.internal:
        nname = node_name(node)
        # st += dec_node_box(name, nname, split=round(threshold[i]))
        gr_node = split_node(node.feature_name(), nname, split=round(node.split()))
        internal.append(gr_node)

        split_viz(node, X_train, y_train, filename=f"/tmp/node{node.id}.svg",
                  target_name=target_name,
                  figsize=figsize,
                  y_range=y_range,
                  show_ylabel=node == shadow_tree.root,
                  showx=True,
                  precision=precision,
                  colors=colors)

    leaves = []
    for node in shadow_tree.leaves:
        regr_leaf_viz(node, y_train, target_name=target_name, filename=f"/tmp/node{node.id}.svg",
                      y_range=y_range, precision=precision)

        if shadow_tree.isclassifier():
            leaves.append( class_leaf_node(node) )
        else:
            leaves.append( regr_leaf_node(node) )

    fromport = ""
    toport = ""
    if fancy and orientation=="TD":
        fromport = ":img:s"
        toport = ":img:n"
    elif fancy:
        fromport = ":img:e"
        toport = ":img:w"
    llabel = '&lt;' if show_edge_labels else ''
    rlabel = '&ge;' if show_edge_labels else ''

    edges = []
    # non leaf edges with > and <=
    for node in shadow_tree.internal:
        nname = node_name(node)
        left_node_name = node_name(node.left)
        if node.left.isleaf():
            left_node_name ='leaf%d' % node.left.id
        right_node_name = node_name(node.right)
        if node.right.isleaf():
            right_node_name ='leaf%d' % node.right.id
        edges.append( f'{nname}{fromport} -> {left_node_name}{toport} [label=<{llabel}>]' )
        edges.append( f'{nname}{fromport} -> {right_node_name}{toport} [label=<{rlabel}>]' )

    newline = "\n\t"
    st = f"""
digraph G {{splines=line;
    nodesep=0.1;
    ranksep={ranksep};
    rankdir={orientation};
    node [margin="0.03" penwidth="0.5" width=.1, height=.1];
    edge [arrowsize=.4 penwidth="0.3"]
    
    {newline.join(internal)}
    {newline.join(edges)}
    {newline.join(leaves)}
    
    subgraph cluster_legend {{
        style=invis;
        legend [penwidth="0.3" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
        {class_legend_html()}
        >]
    }}
}}
    """

    return graphviz.Source(st)


def split_viz(node: ShadowDecTreeNode,
              X: (pd.DataFrame, np.ndarray),
              y: (pd.Series, np.ndarray),
              target_name: str,
              colors : Mapping[int,str],
              filename: str = None,
              showx=True,
              showy=True,
              show_ylabel=True,
              y_range=None,
              figsize: Tuple[Number, Number] = None,
              ticks_fontsize: int = 18,
              label_fontsize: int = 20,
              precision=1):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.tick_params(colors=GREY)

    feature_name = node.feature_name()
    if show_ylabel: showy=True
    if showx:
        ax.set_xlabel(f"{feature_name}@{round(node.split(),precision)}", fontsize=label_fontsize, fontname="Arial", color=GREY)
    else:
        ax.xaxis.set_visible(False)
        ax.set_xticks([])
    if showy:
        ax.set_ylim(y_range)
        if show_ylabel:
            ax.set_ylabel(target_name, fontsize=label_fontsize, fontname="Arial", color=GREY)
    else:
        ax.yaxis.set_visible(False)
        ax.set_yticks([])

    if not showx and not showy:
        ax.axis('off')

    if isinstance(X,pd.DataFrame):
        X = X.values
    if isinstance(y,pd.Series):
        y = y.values

    X = X[:,node.feature()]
    X, y = X[node.samples()], y[node.samples()]

    class_values = node.shadowtree.unique_target_values

    if node.isclassifier():
        fig, ax = plt.subplots(1, 1, figsize=(7.5,3.5))
        ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize, fontname="Arial",
                      color=GREY)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        n_classes = node.shadowtree.nclasses()
        class_names = node.shadowtree.class_names

        X_hist = [X[y==cl] for cl in class_values]
        X_colors = [colors[cl] for cl in class_values]
        class_counts = [len(x) for x in X_hist]
        hist, bins, barcontainers = ax.hist(X_hist,
                                            color=X_colors,
                                            label=class_names)

        ax.set_xticks([round(node.split(),precision)])
        ax.tick_params(direction='out', length=15, width=10, color=GREY, labelsize=label_fontsize)

        # Alter appearance of each bar
        for patch in barcontainers:
            for rect in patch.patches:
                rect.set_linewidth(1.2)
                rect.set_edgecolor(GREY)
    else:
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)

        ax.scatter(X, y, s=3, c=LIGHTBLUE, alpha=1.0)
        left, right = node.split_samples()
        left = y[left]
        right = y[right]
        split = node.split()
        ax.plot([min(X),split],[np.mean(left),np.mean(left)],'--', color=GREY, linewidth=1.6)
        ax.plot([split,split],[min(y),max(y)],'--', color=GREY, linewidth=1.6)
        ax.plot([split,max(X)],[np.mean(right),np.mean(right)],'--', color=GREY, linewidth=1.6)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

def regr_leaf_viz(node : ShadowDecTreeNode,
                  y : (pd.Series,np.ndarray),
                  target_name,
                  filename:str=None,
                  y_range=None,
                  precision=1,
                  figsize:Tuple[Number,Number]=(2.9, 3.0),
                  label_fontsize:int=28,
                  ticks_fontsize: int = 24):
    if isinstance(y,pd.Series):
        y = y.values

    y = y[node.samples()]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plt.subplots_adjust(wspace=5)
    axes[0].set_ylim(y_range)
    axes[0].tick_params(axis='x', which='both', labelsize=label_fontsize, colors=GREY)
    axes[0].tick_params(axis='y', which='both', labelsize=ticks_fontsize, colors=GREY)
    # axes[0].tick_params(axis='both', which='both', labelsize=ticks_fontsize, colors=GREY)

    meanprops = {'linewidth': 1.2, 'linestyle': '-', 'color': 'black'}
    bp = axes[0].boxplot(y, labels=[target_name],
                        notch=False, medianprops={'linewidth': 0}, meanprops=meanprops,
                        widths=[.8], showmeans=True, meanline=True, sym='', patch_artist=True)
    for patch in bp['boxes']:
        patch.set(facecolor=LIGHTBLUE)

    axes[1].yaxis.tick_right()
    axes[1].set_ylim(0, 350)
    axes[1].tick_params(axis='x', which='both', labelsize=label_fontsize, colors=GREY)
    axes[1].tick_params(axis='y', which='both', labelsize=ticks_fontsize, colors=GREY)
    axes[1].bar(0, node.nsamples(), color=LIGHTORANGE, tick_label="n")
    axes[1].axhline(node.nsamples(), color=GREY, linewidth=1.2)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def draw_legend_boxes(shadow_tree, basefilename):
    n_classes = shadow_tree.nclasses()
    class_values = shadow_tree.unique_target_values
    color_values = color_blind_friendly_colors[n_classes]
    colors = {v:color_values[i] for i,v in enumerate(class_values)}

    for i, c in enumerate(class_values):
        draw_colored_box(colors[c], f"{basefilename}{i}.svg")


def draw_legend(shadow_tree, filename):
    "Unused since we can't get accurate size measurement for HTML label on node."
    fig, ax = plt.subplots(1, 1, figsize=(.1,.1))
    boxes = []

    n_classes = shadow_tree.nclasses()
    class_values = shadow_tree.unique_target_values
    class_names = shadow_tree.class_names
    color_values = color_blind_friendly_colors[n_classes]
    colors = {v:color_values[i] for i,v in enumerate(class_values)}

    for i, c in enumerate(class_values):
        b = patches.Rectangle((0, 0), 0, 0, linewidth=1.2, edgecolor='grey',
                                 facecolor=colors[c], label=class_names[c])
        boxes.append(b)
    ax.legend(handles=boxes,
              frameon=False,
              loc='center',
              edgecolor='red',
              fontsize=18)

    ax.set_xlim(0, 0.01)
    ax.set_ylim(0, 0.01)
    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def draw_colored_box(color,filename):
    fig, ax = plt.subplots(1, 1, figsize=(.65, .5))

    box1 = patches.Rectangle((0, 0), 2, 1, linewidth=1.2, edgecolor='grey',
                             facecolor=color)

    ax.add_patch(box1)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def prop_size(n, counts, output_range = (0.00, 0.3)):
    min_samples = min(counts)
    max_samples = max(counts)
    sample_count_range = max_samples - min_samples


    if sample_count_range>0:
        zero_to_one = (n - min_samples) / sample_count_range
        return zero_to_one * (output_range[1] - output_range[0]) + output_range[0]
    else:
        return output_range[0]


def boston():
    regr = tree.DecisionTreeRegressor(max_depth=3, random_state=666)
    boston = load_boston()

    data = pd.DataFrame(boston.data)
    data.columns =boston.feature_names

    regr = regr.fit(data, boston.target)

    # st = dectreeviz(regr.tree_, data, boston.target)
    st = dtreeviz(regr, data, boston.target, target_name='price',
                  feature_names=data.columns, orientation="TD",
                  show_edge_labels=False,
                  fancy=True)

    with open("/tmp/t3.dot", "w") as f:
        f.write(st.source)

    return st

def iris():
    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=666)
    iris = load_iris()

    #print(iris.data.shape, iris.target.shape)

    data = pd.DataFrame(iris.data)
    data.columns = iris.feature_names

    clf = clf.fit(data, iris.target)

    # st = dectreeviz(clf.tree_, data, boston.target)
    st = dtreeviz(clf, data, iris.target,target_name='variety',
                  feature_names=data.columns, orientation="TD",
                  class_names=["setosa", "versicolor", "virginica"], # 0,1,2 targets
                  fancy=True, show_edge_labels=False)
    #print(st)

    with open("/tmp/t3.dot", "w") as f:
        f.write(st.source)

    #print(clf.tree_.value)
    return st

st = iris()
#st = boston()
st.view()
#
