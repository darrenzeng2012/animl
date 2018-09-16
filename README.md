# animl

A python machine learning library for scikit-learn decision tree visualization and model interpretation. 

Decision trees are the fundamental building block of gradient boosting machines and Random Forests™, probably the two most popular machine learning models for structured data. Visualizing decision trees is a tremendous aid when learning these models and later, in practice, when interpreting models. Unfortunately, current visualization packages are rudimentary and not immediately helpful to the novice. For example, we couldn't find a library that could visualize how decision nodes split up feature space. Our library could be the first. It also appears uncommon for libraries to support visualizing a specific feature vector as it weaves down through a tree's decision nodes, as we could only find one image showing this (but we didn't exhaustively look through library APIs).

The visualizations are inspired by an educational animiation by [R2D3](http://www.r2d3.us/); [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/). With animl, you can visualize how the feature space is split up at decision nodes, how the training samples get ditributed in leaf nodes and how the tree makes prediction for a new observation. These operations are very critical to grasp for someone who wants to understand how classfication or regression decision tree works. 

There are lots of intricaties in the plots that Terence had to obsess over during the contruction of the library. See [How to visualize decision tree models](https://www.google.com/) for deeper discussion of our dicision tree visualization tool and how we have chosen to visualize the feature-target space of a decision tree. 


## Requirements

Needs graphviz/dot lib.

```bash
pip install graphviz
```

And, on latest mac versions, you need:

```bash
brew install graphviz --with-librsvg --with-app --with-pango
```


## Usage and examples

* `dtree`: Main function to create decision tree visualization. Given a decision tree regressor or classifier, creates and returns a tree visualization using the graphviz (DOT) language.

* **Regression decision tree with default settings**: 


```bash
regr = tree.DecisionTreeRegressor(max_depth=2)
boston = load_boston()
regr = regr.fit(boston.data, boston.target)

st = dtreeviz(regr, boston.data, boston.target, target_name='price',
              feature_names=boston.feature_names)
              
g = graphviz.Source(st, format='pdf')
g.render(directory=".", filename="boston.pdf", view=False, cleanup=True)
g.view()              
```
  
<img src=testing/samples/boston-TD-2.png width=320 height=320>
  
  
* **Classification decision tree with default settings**:
```bash
classifier = tree.DecisionTreeClassifier(max_depth=2)  # limit depth of tree
iris = load_iris()
regr = classifier.fit(iris.data, iris.target)

# need class_names for classifier
st = dtreeviz(classifier, iris.data, iris.target, target_name='variety',
              feature_names=iris.feature_names, class_names=["setosa", "versicolor", "virginica"] )
              
g = graphviz.Source(st, format='pdf')
g.render(directory=".", filename="iris.pdf", view=False, cleanup=True)
g.view()  
```

<img src=testing/samples/iris-TD-2.png width=320 height=320>


