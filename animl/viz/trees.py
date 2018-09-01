import numpy as np
import pandas as pd
import graphviz
from numpy.distutils.system_info import f2py_info
from sklearn import tree
import matplotlib.pyplot as plt
from animl.trees import *
from numbers import Number
import matplotlib.patches as patches
from scipy import stats
from sklearn.neighbors import KernelDensity
import tempfile

YELLOW = "#fefecd" # "#fbfbd0" # "#FBFEB0"
BLUE = "#D9E6F5"
GREEN = "#cfe2d4"
DARKBLUE = '#313695'
DARKGREEN = '#006400'
LIGHTORANGE = '#fee090'
LIGHTBLUE = '#a6bddb'
GREY = '#444443'

HIGHLIGHT_COLOR = '#D67C03'

# How many bins should we have based upon number of classes
NUM_BINS = [0, 0, 10, 9, 8, 6, 6, 6, 5, 5, 5]
          # 0, 1, 2,  3, 4, 5, 6, 7, 8, 9, 10

color_blind_friendly_colors = [
    None, # 0 classes
    None, # 1 class
    ["#fefecd","#a1dab4"], # 2 classes
    ["#fefecd","#D9E6F5",'#a1dab4'], # 3 classes
    ["#fefecd","#D9E6F5",'#a1dab4','#fee090'], # 4
    ["#fefecd","#D9E6F5",'#a1dab4','#41b6c4','#fee090'], # 5
    ["#fefecd",'#c7e9b4','#41b6c4','#2c7fb8','#fee090','#f46d43'], # 6
    ["#fefecd",'#c7e9b4','#7fcdbb','#41b6c4','#225ea8','#fdae61','#f46d43'], # 7
    ["#fefecd",'#edf8b1','#c7e9b4','#7fcdbb','#1d91c0','#225ea8','#fdae61','#f46d43'], # 8
    ["#fefecd",'#c7e9b4','#41b6c4','#74add1','#4575b4','#313695','#fee090','#fdae61','#f46d43'], # 9
    ["#fefecd",'#c7e9b4','#41b6c4','#74add1','#4575b4','#313695','#fee090','#fdae61','#f46d43','#d73027'] # 10
]

def dtreeviz(tree_model : (tree.DecisionTreeRegressor,tree.DecisionTreeClassifier),
             X_train : (pd.DataFrame, np.ndarray),
             y_train : (pd.Series, np.ndarray),
             feature_names : List[str],
             target_name : str,
             class_names : (Mapping[Number,str],List[str]) = None, # required if classifier
             precision  : int = 1,
             orientation : ('TD','LR') ="TD",
             show_root_edge_labels : bool = True,
             fancy : bool = True,
             histtype: ('bar', 'barstacked') = 'barstacked',
             highlight_path : List[int] = [])\
        -> str:
    """
    Given a decision tree regressor or classifier, create and return a tree visualization
    using the graphviz (DOT) language.

    :param tree_model: A DecisionTreeRegressor or DecisionTreeClassifier that has been
                       fit to X_train, y_train.
    :param X_train: A data frame or 2-D matrix of feature vectors used to train the model.
    :param y_train: A pandas Series or 1-D vector with target values or classes.
    :param feature_names: A list of the feature names.
    :param target_name: The name of the target variable.
    :param class_names: [For classifiers] A dictionary or list of strings mapping class
                        value with class name.
    :param precision: When displaying floating-point numbers, how many digits to display
                      after the decimal point.
    :param orientation:  Is the tree top down, "TD", or left to right, "LR"?
    :param show_root_edge_labels: Include < and >= on the edges emanating from the root?
    :param fancy:
    :param histtype: [For classifiers] Either 'bar' or 'barstacked' to indicate
                     histogram type. We find that 'barstacked' looks great up to about.
                     four classes.
    :param highlight_path: A list of node IDs to highlight, default is []
    :type highlight_path: List[int]
    :return: A string in graphviz DOT language that describes the decision tree.
    """
    def round(v,ndigits=precision):
        return format(v, '.' + str(ndigits) + 'f')

    def node_name(node : ShadowDecTreeNode) -> str:
        if node.feature_name() is None:
            return f"node{node.id}"
        node_name = ''.join(c for c in node.feature_name() if c not in string.punctuation)+str(node.id)
        node_name = re.sub("["+string.punctuation+string.whitespace+"]", '_', node_name)
        return node_name

    def split_node(name, node_name, split):
        img_shape = get_SVG_shape(f"{tmp}/node{node.id}.svg")
        if fancy:
            html = f"""<table border="0">
            <tr>
                    <td port="img" fixedsize="true" width="{img_shape[0]}" height="{img_shape[1]}"><img src="{tmp}/node{node.id}.svg"/></td>
            </tr>
            </table>"""
        else:
            html = f"""<font face="Helvetica" color="#444443" point-size="12">{name}@{split}</font>"""
        if node.id in highlight_path:
            gr_node = f'{node_name} [margin="0" shape=box penwidth=".5" color="{HIGHLIGHT_COLOR}" style="dashed" label=<{html}>]'
        else:
            gr_node = f'{node_name} [margin="0" shape=none label=<{html}>]'
        return gr_node


    def regr_leaf_node(node, label_fontsize: int = 12):
        img_shape = get_SVG_shape(f"{tmp}/node{node.id}.svg")
        value = node.prediction()
        if fancy:
            html = f"""<table border="0">
            <tr>
                    <td port="img" fixedsize="true" width="{img_shape[0]}" height="{img_shape[1]}"><img src="{tmp}/node{node.id}.svg"/></td>
            </tr>
            </table>"""
            if node.id in highlight_path:
                return f'leaf{node.id} [margin="0" shape=box penwidth=".5" color="{HIGHLIGHT_COLOR}" style="dashed" label=<{html}>]'
            else:
                return f'leaf{node.id} [margin="0" shape=plain label=<{html}>]'
        else:
            width = prop_size(node.nsamples(),
                              counts=shadow_tree.leaf_sample_counts(),
                              output_range=(1.01,2.7))
            width = np.sqrt(np.log(width)) # see small n diffs visually
            gr = f'leaf{node.id} [fixedsize="true" width="{width}" style=filled fillcolor="{YELLOW}" shape=circle label=""]'

            if orientation == 'TD':
                labeldistance = "1.2"
            else:
                labeldistance = "2.2"
            # label = f'<font face="Helvetica" color="{GREY}" point-size="{label_fontsize}">n={node.nsamples()}</font>'
            label = f"""<table border="0" CELLPADDING="0" CELLBORDER="0" CELLSPACING="0">
            <tr>
                    <td align="center" CELLPADDING="0" CELLSPACING="0"><font face="Helvetica" color="#444443" point-size="11">{target_name}={round(value)}</font></td>
            </tr>
            <tr>
                    <td align="center" CELLPADDING="0" CELLSPACING="0"><font face="Helvetica" color="#444443" point-size="11">n={node.nsamples()}</font></td>
            </tr>
            </table>
            """
            spacer_width = .15 * (1 / width) # smaller nodes need bigger space for labels
            annot = f"""
               leaf{node.id}_annot [shape=none width="{spacer_width}" label=""]
               leaf{node.id}_spacer [shape=none label=""]
               leaf{node.id} -> leaf{node.id}_annot [penwidth=0 arrowsize=0 labeldistance="{labeldistance}" labelangle="0" taillabel=<{label}>]
                {{
                    rank=same;
                    leaf{node.id} -> leaf{node.id}_spacer [style=invis]
                }}
            """
            return gr + annot

    def class_leaf_node(node, label_fontsize: int = 12):
        img_shape = get_SVG_shape(f"{tmp}/node{node.id}.svg")
        html = f"""<table border="0" CELLBORDER="0">
        <tr>
                <td port="img" fixedsize="true" width="{img_shape[0]}" height="{img_shape[1]}"><img src="{tmp}/node{node.id}.svg"/></td>
        </tr>
        </table>"""
        if node.id in highlight_path:
            return f'leaf{node.id} [margin="0" shape=box penwidth=".5" color="{HIGHLIGHT_COLOR}" style="dashed" label=<{html}>]'
        else:
            return f'leaf{node.id} [margin="0" shape=plain label=<{html}>]'

    def class_legend_html(label_fontsize: int = 12):
        elements = []
        for i,cl in enumerate(class_values):
            html = f"""
            <tr>
                <td border="0" cellspacing="0" cellpadding="0"><img src="{tmp}/legend{i}.svg"/></td>
                <td align="left"><font face="Helvetica" color="{GREY}" point-size="{label_fontsize}">{class_names[cl]}</font></td>
            </tr>
            """
            elements.append(html)
        return f"""
        <table border="0" cellspacing="0" cellpadding="0">
        <tr>
            <td border="0" colspan="2"><font face="Helvetica" color="{GREY}" point-size="{label_fontsize}"><b>{target_name}</b></font></td>
        </tr>
        {''.join(elements)}
        </table>
        """

    def class_legend_gr():
        if not shadow_tree.isclassifier():
            return ""
        return f"""
            subgraph cluster_legend {{
                style=invis;
                legend [penwidth="0.3" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
                {class_legend_html()}
                >]
            }}
            """

    ranksep = ".22"
    if orientation=="TD":
        ranksep = ".4"

    tmp = tempfile.gettempdir()

    shadow_tree = ShadowDecTree(tree_model, X_train, y_train,
                                feature_names=feature_names, class_names=class_names)

    n_classes = shadow_tree.nclasses()
    color_values = color_blind_friendly_colors[n_classes]

    # Fix the mapping from target value to color for entire tree
    colors = None
    if shadow_tree.isclassifier():
        class_values = shadow_tree.unique_target_values
        colors = {v:color_values[i] for i,v in enumerate(class_values)}

    y_range = (min(y_train)*1.03, max(y_train)*1.03) # same y axis for all

    if shadow_tree.isclassifier():
        draw_legend_boxes(shadow_tree, f"{tmp}/legend")

    if isinstance(X_train,pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train,pd.Series):
        y_train = y_train.values

    # Find max height (count) for any bar in any node
    if shadow_tree.isclassifier():
        nbins = get_num_bins(histtype, n_classes)
        node_heights = shadow_tree.get_split_node_heights(X_train, y_train, nbins=nbins)

    internal = []
    for node in shadow_tree.internal:
        if fancy:
            if shadow_tree.isclassifier():
                class_split_viz(node, X_train, y_train,
                                filename=f"{tmp}/node{node.id}.svg",
                                precision=precision,
                                colors=colors,
                                histtype=histtype,
                                node_heights=node_heights)
            else:
                regr_split_viz(node, X_train, y_train,
                               filename=f"{tmp}/node{node.id}.svg",
                               target_name=target_name,
                               y_range=y_range,
                               precision=precision)

        nname = node_name(node)
        gr_node = split_node(node.feature_name(), nname, split=round(node.split()))
        internal.append(gr_node)

    leaves = []
    for node in shadow_tree.leaves:
        if shadow_tree.isclassifier():
            class_leaf_viz(node, colors=color_values,
                           filename=f"{tmp}/node{node.id}.svg")
            leaves.append( class_leaf_node(node) )
        else:
            if fancy:
                regr_leaf_viz(node, y_train, target_name=target_name,
                              filename=f"{tmp}/node{node.id}.svg",
                              y_range=y_range, precision=precision)
            leaves.append( regr_leaf_node(node) )

    fromport = ""
    toport = ""
    if fancy and orientation=="TD":
        fromport = ":img:s"
        toport = ":img:n"
    elif fancy:
        fromport = ":img:e"
        toport = ":img:w"

    if shadow_tree.isclassifier():
        fromport = toport = ""

    show_edge_labels = False
    all_llabel = '&lt;' if show_edge_labels else ''
    all_rlabel = '&ge;' if show_edge_labels else ''
    root_llabel = '&lt;' if show_root_edge_labels else ''
    root_rlabel = '&ge;' if show_root_edge_labels else ''

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
        llabel = all_llabel
        rlabel = all_rlabel
        if node==shadow_tree.root:
            llabel = root_llabel
            rlabel = root_rlabel
        edges.append( f'{nname}{fromport} -> {left_node_name}{toport} [label=<{llabel}>]' )
        edges.append( f'{nname}{fromport} -> {right_node_name}{toport} [label=<{rlabel}>]' )
        edges.append(f"""
        {{
            rank=same;
            {left_node_name} -> {right_node_name} [style=invis]
        }}
        """)

    newline = "\n\t"
    st = f"""
digraph G {{
    splines=line;
    nodesep=0.1;
    ranksep={ranksep};
    rankdir={orientation};
    node [margin="0.03" penwidth="0.5" width=.1, height=.1];
    edge [arrowsize=.4 penwidth="0.3"]
    
    {newline.join(internal)}
    {newline.join(edges)}
    {newline.join(leaves)}
    
    {class_legend_gr()}
}}
    """

    return st


def class_split_viz(node: ShadowDecTreeNode,
                    X: np.ndarray,
                    y: np.ndarray,
                    colors: Mapping[int, str],
                    node_heights,
                    filename: str = None,
                    ticks_fontsize: int = 8,
                    label_fontsize: int = 9,
                    precision=1,
                    histtype: ('bar', 'barstacked') = 'barstacked'):
    height_range = (.5, 1.5)
    h = prop_size(n=node_heights[node.id], counts=node_heights.values(), output_range=height_range)
    figsize=(3.3, h)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    feature_name = node.feature_name()
    ax.set_xlabel(f"{feature_name}@{round(node.split(),precision)}", fontsize=label_fontsize, fontname="Arial", color=GREY)

    # Get X, y data for all samples associated with this node.
    X_feature = X[:,node.feature()]
    X_feature, y = X_feature[node.samples()], y[node.samples()]

    n_classes = node.shadow_tree.nclasses()
    nbins = get_num_bins(histtype, n_classes)
    overall_feature_range = (np.min(X[:,node.feature()]), np.max(X[:,node.feature()]))

    ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize, fontname="Arial",
                  color=GREY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)

    class_names = node.shadow_tree.class_names

    r = overall_feature_range[1]-overall_feature_range[0]

    class_values = node.shadow_tree.unique_target_values
    X_hist = [X_feature[y==cl] for cl in class_values]
    X_colors = [colors[cl] for cl in class_values]
    binwidth = r / nbins

    hist, bins, barcontainers = ax.hist(X_hist,
                                        color=X_colors,
                                        align='mid',
                                        histtype=histtype,
                                        bins=np.arange(overall_feature_range[0],overall_feature_range[1] + binwidth, binwidth),
                                        label=class_names)

    ax.set_xlim(*overall_feature_range)
    ax.set_xticks(overall_feature_range)
    ax.set_yticks([0,max([max(h) for h in hist])])
    ax.tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = xmax-xmin
    yr = ymax-ymin
    hr = h / (height_range[1]-height_range[0])
    th = yr*.15 * 1/hr # convert to graph coordinates (ugh)
    tw = xr*.018
    tipy = -0.1 * yr *.15 * 1/hr
    tria = np.array([[node.split(), tipy], [node.split() - tw, -th], [node.split() + tw, -th]])
    t = patches.Polygon(tria, facecolor='orange')
    t.set_clip_on(False)
    ax.add_patch(t)
    if (node.split()-overall_feature_range[0]) >= .5*r:
        ax.text(node.split() - tw, -1.2*th,
                f"{round(node.split(),1)}",
                horizontalalignment='right',
                fontsize=ticks_fontsize, color=GREY)
    else:
        ax.text(node.split() + tw, -1.2*th,
                f"{round(node.split(),1)}",
                horizontalalignment='left',
                fontsize=ticks_fontsize, color=GREY)


    # Alter appearance of each bar
    for patch in barcontainers:
        for rect in patch.patches:
            rect.set_linewidth(.5)
            rect.set_edgecolor(GREY)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def class_leaf_viz(node : ShadowDecTreeNode,
                   colors : List[str],
                   filename: str):
    size = prop_size(node.nsamples(), counts=node.shadow_tree.leaf_sample_counts(),
                     output_range=(1.01, 2.7))
    # we visually need n=1 and n=9 to appear different but diff between 300 and 400 is no big deal
    size = np.sqrt(np.log(size))
    draw_piechart(node.class_counts(), size=size, colors=colors, filename=filename, label=f"n={node.nsamples()}")


def regr_split_viz(node: ShadowDecTreeNode,
                   X: np.ndarray,
                   y: np.ndarray,
                   target_name: str,
                   filename: str = None,
                   y_range=None,
                   ticks_fontsize: int = 8,
                   label_fontsize: int = 9,
                   precision=1):
    figsize = (2.5, 1.1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.tick_params(colors=GREY)

    feature_name = node.feature_name()
    ticklabelpad = plt.rcParams['xtick.major.pad']
    ax.annotate(f"{feature_name}@{round(node.split(),precision)}",
                xy=(.5, 0), xytext=(.5, -2*ticklabelpad), ha='center', va='top',
                xycoords='axes fraction', textcoords='offset points',
                fontsize = label_fontsize, fontname = "Arial", color = GREY)

    ax.set_ylim(y_range)
    if node==node.shadow_tree.root:
        ax.set_ylabel(target_name, fontsize=label_fontsize, fontname="Arial", color=GREY)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)
    ax.tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)

    # Get X, y data for all samples associated with this node.
    X_feature = X[:,node.feature()]
    X_feature, y = X_feature[node.samples()], y[node.samples()]

    overall_feature_range = (np.min(X[:,node.feature()]), np.max(X[:,node.feature()]))
    ax.set_xlim(*overall_feature_range)
    ax.set_xticks(overall_feature_range)


    ax.scatter(X_feature, y, s=5, c='#225ea8', alpha=.4)
    left, right = node.split_samples()
    left = y[left]
    right = y[right]
    split = node.split()
    ax.plot([overall_feature_range[0],split],[np.mean(left),np.mean(left)],'--', color=GREY, linewidth=.5)
    ax.plot([split,split],[min(y),max(y)],'--', color=GREY, linewidth=.5)
    ax.plot([split,overall_feature_range[1]],[np.mean(right),np.mean(right)],'--', color=GREY, linewidth=.5)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def regr_leaf_viz(node : ShadowDecTreeNode,
                  y : (pd.Series,np.ndarray),
                  target_name,
                  filename:str=None,
                  y_range=None,
                  precision=1,
                  label_fontsize:int=9,
                  ticks_fontsize: int = 8):
    samples = node.samples()
    y = y[samples]

    figsize = (.8,1.2)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.tick_params(colors=GREY)

    m = np.mean(y)

    ax.set_ylim(y_range)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.set_xticks([])
    # ax.set_yticks(y_range)

    ticklabelpad = plt.rcParams['xtick.major.pad']
    ax.annotate(f"{target_name}={round(m,precision)}\nn={len(y)}",
                xy=(.5, 0), xytext=(.5, -.5*ticklabelpad), ha='center', va='top',
                xycoords='axes fraction', textcoords='offset points',
                fontsize = label_fontsize, fontname = "Arial", color = GREY)

    ax.tick_params(axis='y', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)

    mu = .5
    sigma = .15
    X = np.random.normal(mu, sigma, size=len(y))
    ax.set_xlim(0, 1)
    alpha = .25

    ax.scatter(X, y, s=5, c='#225ea8', alpha=alpha)
    ax.plot([0,len(node.samples())],[m,m],'--', color=GREY, linewidth=.5)

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

def draw_piechart(counts,size,colors,filename,label=None):
    n_nonzero = np.count_nonzero(counts)
    i = np.nonzero(counts)[0][0]
    if n_nonzero==1:
        counts = [counts[i]]
        colors = [colors[i]]
    tweak = size * .01
    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    ax.axis('equal')
    # ax.set_xlim(0 - tweak, size + tweak)
    # ax.set_ylim(0 - tweak, size + tweak)
    ax.set_xlim(0, size-10*tweak)
    ax.set_ylim(0, size-10*tweak)
    # frame=True needed for some reason to fit pie properly (ugh)
    # had to tweak the crap out of this to get tight box around piechart :(
    wedges, _ = ax.pie(counts, center=(size/2-6*tweak,size/2-6*tweak), radius=size/2, colors=colors, shadow=False, frame=True)
    for w in wedges:
        w.set_linewidth(.5)
        w.set_edgecolor(GREY)

    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if label is not None:
        ax.text(size/2-6*tweak, -10*tweak, label,
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=9, color=GREY, fontname="Arial")

    # plt.tight_layout()
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


def get_num_bins(histtype, n_classes):
    bins = NUM_BINS[n_classes]
    if histtype == 'barstacked':
        bins *= 2
    return bins


def get_SVG_shape(filename):
    """
    Sample line from SVG file from which we can get w,h:
    <svg height="122.511795pt" version="1.1" viewBox="0 0 451.265312 122.511795"
         width="451.265312pt" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="...">
    """
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("<svg "):
                args = line[len("<svg "):].split()
                d = {}
                for arg in args:
                    a = arg.split('=')
                    if len(a)==2:
                        d[a[0]] = a[1].strip('"').strip('pt')
                return float(d['width']), float(d['height'])

