import numpy as np
import pandas as pd
import graphviz
from numpy.distutils.system_info import f2py_info
from sklearn import tree
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, load_breast_cancer, load_diabetes, fetch_mldata
from matplotlib.figure import figaspect
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
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

max_class_colors = len(color_blind_friendly_colors)-1

def dtreeviz(tree_model, X_train, y_train, feature_names, target_name, class_names=None,
             precision=1, orientation="TD", show_root_edge_labels=True,
             fancy=True,
             histtype: ('bar', 'barstacked') = 'barstacked') -> str:
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
        gr_node = f'{node_name} [margin="0" shape=none label=<{html}>]'
        return gr_node


    def regr_leaf_node(node):
        img_shape = get_SVG_shape(f"{tmp}/node{node.id}.svg")
        value = node.prediction()
        if fancy:
            html = f"""<table border="0" CELLPADDING="0" CELLBORDER="0" CELLSPACING="0">
            <tr>
                    <td colspan="3" port="img" fixedsize="true" width="{img_shape[0]}" height="{img_shape[1]}"><img src="{tmp}/node{node.id}.svg"/></td>
            </tr>
            </table>"""
            return f'leaf{node.id} [margin="0" shape=plain label=<{html}>]'
        else:
            margin = prop_size(node.nsamples(),
                               counts=shadow_tree.leaf_sample_counts())
            html = f"""<font face="Helvetica" color="#444443" point-size="11">{target_name}<br/>{round(value)}</font>"""
            return f'leaf{node.id} [margin="{margin}" style=filled fillcolor="{YELLOW}" shape=circle label=<{html}>]'

    def class_leaf_node(node, label_fontsize: int = 12):
        counts = node.class_counts()
        predicted_class = np.argmax(counts)
        predicted = predicted_class
        if class_names is not None:
            predicted = class_names[predicted_class]
        n_nonzero = np.count_nonzero(counts)
        ratios = counts / node.nsamples()  # convert counts to ratios totalling 1.0
        ratios = [round(r, 3) for r in ratios]
        color_spec = ["{c};{r}".format(c=color_values[i], r=r) for i, r in
                      enumerate(ratios)]
        color_spec = ':'.join(color_spec)
        if n_classes > max_class_colors:
            color_spec = LIGHTBLUE
        if n_nonzero==1: # make pure
            i = np.nonzero(counts)[0][0]
            color_spec = color_values[i]
        #width = prop_size(node.nsamples(), counts = shadow_tree.leaf_sample_counts(), output_range=(.2,.8))
        width = prop_size(node.nsamples(), counts = shadow_tree.leaf_sample_counts(), output_range=(.15,.85))
        style = 'wedged' if n_classes <= max_class_colors and n_nonzero>1 else 'filled'
        adjust = ""
        labelangle = "0"
        if style=='wedged':
            adjust = "<br/>&nbsp;"
            if orientation=='TD':
                labeldistance="1.2"
            else:
                labelangle = "-20"
                labeldistance = "1.4"
        else:
            if orientation=='TD':
                labeldistance=".6"
            else:
                labeldistance = "1.3"
        label = f'<font face="Helvetica" color="{GREY}" point-size="{label_fontsize}">n={node.nsamples()}{adjust}</font>'
        gr = f'leaf{node.id} [fixedwidth="true" width="{width}" style={style} fillcolor="{color_spec}" shape=circle label=""]'
        annot = f"""
           leaf{node.id}_annot [shape=none label=""]
           leaf{node.id} -> leaf{node.id}_annot [penwidth=0 arrowsize=0 labeldistance="{labeldistance}" labelangle="{labelangle}" taillabel=<{label}>]
        """
        return gr + annot

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

    #figsize = (4.5, 2)

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
        if shadow_tree.isclassifier():
            class_split_viz(node, X_train, y_train,
                            filename=f"{tmp}/node{node.id}.svg",
                            target_name=target_name,
                            precision=precision,
                            colors=colors,
                            histtype=histtype,
                            node_heights=node_heights)
        else:
            regr_split_viz(node, X_train, y_train,
                           filename=f"{tmp}/node{node.id}.svg",
                           target_name=target_name,
                           y_range=y_range,
                           show_ylabel=node == shadow_tree.root,
                           showx=True,
                           precision=precision,
                           colors=colors)

        nname = node_name(node)
        gr_node = split_node(node.feature_name(), nname, split=round(node.split()))
        internal.append(gr_node)

    leaves = []
    for node in shadow_tree.leaves:
        regr_leaf_viz(node, y_train, target_name=target_name, filename=f"{tmp}/node{node.id}.svg",
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
digraph G {{splines=line;
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
                    target_name: str,
                    colors: Mapping[int, str],
                    node_heights,
                    filename: str = None,
                    figsize: Tuple[Number, Number] = None,
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

    n_classes = node.shadowtree.nclasses()
    nbins = get_num_bins(histtype, n_classes)
    overall_feature_range = (np.min(X[:,node.feature()]), np.max(X[:,node.feature()]))

    ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize, fontname="Arial",
                  color=GREY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)

    class_names = node.shadowtree.class_names

    r = overall_feature_range[1]-overall_feature_range[0]

    class_values = node.shadowtree.unique_target_values
    X_hist = [X_feature[y==cl] for cl in class_values]
    X_colors = [colors[cl] for cl in class_values]
    binwidth = r / nbins
    # print(f"{nbins} bins, binwidth for feature {node.feature_name()} is {binwidth}")
    # print(np.arange(overall_feature_range[0], overall_feature_range[1] + binwidth,
    #                  binwidth))

    hist, bins, barcontainers = ax.hist(X_hist,
                                        color=X_colors,
                                        align='mid',
                                        histtype=histtype,
                                        bins=np.arange(overall_feature_range[0],overall_feature_range[1] + binwidth, binwidth),
                                        label=class_names)

    ax.set_xlim(*overall_feature_range)
    ax.set_xticks(overall_feature_range)
    ax.set_yticks([0,max([max(h) for h in hist])])
    #ax.set_xticks(np.arange(*feature_range), (feature_range[1]-feature_range[0])/10.0)
    ax.tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)
    # ax.tick_params(direction='out', width=.3, colors=GREY, labelsize=40)
    # ax.tick_params(colors=GREY)

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


def regr_split_viz(node: ShadowDecTreeNode,
                   X: np.ndarray,
                   y: np.ndarray,
                   target_name: str,
                   colors: Mapping[int, str],
                   filename: str = None,
                   showx=True,
                   showy=True,
                   show_ylabel=True,
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
    if node==node.shadowtree.root:
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


def kde_class_split_viz(node: ShadowDecTreeNode,
                        X: (pd.DataFrame, np.ndarray),
                        y: (pd.Series, np.ndarray),
                        colors: Mapping[int, str],
                        feature_name,
                        label_fontsize: int = 20,
                        precision=1):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5))
    ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize, fontname="Arial",
                  color=GREY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    n_classes = node.shadowtree.nclasses()

    class_values = node.shadowtree.unique_target_values
    X_hist = [X[y==cl] for cl in class_values]
    X_colors = [colors[cl] for cl in class_values]

    i_splitvar = node.feature()
    splitval = node.split()
    for cl in range(n_classes):
        X = X_hist[cl]
        if len(X)==1:
            X_hist[cl] = np.array([X[0],X[0]])
    ranges = {cl:(np.min(X_hist[cl]), np.max(X_hist[cl]))
              for cl in range(n_classes) if len(X_hist[cl])>1}

    use_stats = True
    for cl in range(n_classes):
        n = len(X_hist[cl])
        if cl not in ranges:
            continue
        else:
            r = ranges[cl]
        x_grid = np.linspace(r[0] - r[0] * .25, r[1] * 1.25, 1000)
        if use_stats:
            d = 1
            bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))
            bw = n ** (-1. / (d + 4))
            kernel = stats.gaussian_kde(X_hist[cl], bw_method=bw)
            heights = kernel.evaluate(x_grid)
        else:
            d = 1
            bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))
            bw /= 6
            kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X.reshape(-1, 1))
            log_dens = kde.score_samples(x_grid.reshape(-1, 1))
            heights = np.exp(log_dens)
        #     print(heights)
        plt.plot(x_grid, heights, linewidth=.5, color=GREY)
        plt.fill(x_grid, heights, alpha=.6, color=X_colors[cl])
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = xmax-xmin
    yr = ymax-ymin
    th = yr*.05
    tw = xr*.02
    tria = np.array([[splitval, 0], [splitval - tw, -th], [splitval + tw, -th]])
    t = patches.Polygon(tria, linewidth=.5, edgecolor=GREY,
                        facecolor=GREY)
    t.set_clip_on(False)
    ax.add_patch(t)
    plt.ylim(0.0, ymax)
    ax.set_xticks([])
    ax.yaxis.set_visible(False)


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

    figsize = (1.6,.8)
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
    ax.annotate(f"{target_name}={round(m,precision)}",
                xy=(.5, 0), xytext=(.5, -.5*ticklabelpad), ha='center', va='top',
                xycoords='axes fraction', textcoords='offset points',
                fontsize = label_fontsize, fontname = "Arial", color = GREY)

    ax.tick_params(axis='y', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)

    # Get X, y data for all samples associated with this node.
    X = range(0,len(node.samples()))

    ax.scatter(X, y, s=5, c='#225ea8', alpha=.4)
    left, right = node.split_samples()
    left = y[left]
    right = y[right]
    ax.plot([0,len(node.samples())],[m,m],'--', color=GREY, linewidth=.5)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def old_regr_leaf_viz(node : ShadowDecTreeNode,
                  y : (pd.Series,np.ndarray),
                  target_name,
                  filename:str=None,
                  y_range=None,
                  precision=1,
                  figsize:Tuple[Number,Number]=(1.35, 1.4),
                  label_fontsize:int=9,
                  ticks_fontsize: int = 7):
    samples = node.samples()
    y = y[samples]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    # plt.subplots_adjust(wspace=5)
    axes[0].set_ylim(y_range)
    axes[0].tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)
    axes[0].spines['top'].set_linewidth(.3)
    axes[0].spines['right'].set_linewidth(.3)
    axes[0].spines['left'].set_linewidth(.3)
    axes[0].spines['bottom'].set_linewidth(.3)

    meanprops = {'linewidth': .5, 'linestyle': '-', 'color': 'black'}
    bp = axes[0].boxplot(y, labels=[target_name],
                        notch=False, medianprops={'linewidth': 0}, meanprops=meanprops,
                        widths=[.8], showmeans=True, meanline=True, sym='', patch_artist=True)
    for patch in bp['boxes']:
        patch.set(facecolor='#225ea8', alpha=.6)

    axes[1].yaxis.tick_right()
    axes[1].set_ylim(0, len(node.shadowtree.X_train))
    axes[1].bar(0, node.nsamples(), color=LIGHTORANGE, tick_label="n")
    axes[1].axhline(node.nsamples(), color=GREY, linewidth=.5)
    axes[1].tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)
    axes[1].spines['top'].set_linewidth(.3)
    axes[1].spines['right'].set_linewidth(.3)
    axes[1].spines['left'].set_linewidth(.3)
    axes[1].spines['bottom'].set_linewidth(.3)

    for ax in axes:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(label_fontsize)

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

